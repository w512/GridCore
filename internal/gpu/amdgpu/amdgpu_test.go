package amdgpu

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/w512/gridcore/internal/gpu"
)

const (
	w7800 = "0000:03:00.0" // Radeon PRO W7800 32 GB
	igpu  = "0000:7c:00.0" // the Ryzen's integrated Radeon
)

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// usePCIIDs points the name lookup at test files for the rest of the test.
func usePCIIDs(t *testing.T, paths ...string) {
	old := pciIDsFiles
	pciIDsFiles = paths
	t.Cleanup(func() { pciIDsFiles = old })
}

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// fakeSys is a /sys with the amdgpu driver bound to the given devices,
// each with files name -> content relative to its PCI device directory.
func fakeSys(t *testing.T, devices map[string]map[string]string) string {
	t.Helper()
	root := t.TempDir()
	drv := filepath.Join(root, "bus", "pci", "drivers", "amdgpu")
	for _, f := range []string{"bind", "unbind", "new_id", "remove_id", "uevent"} {
		write(t, filepath.Join(drv, f), "")
	}
	if err := os.Symlink("../../../../module/amdgpu", filepath.Join(drv, "module")); err != nil {
		t.Fatal(err)
	}
	for addr, files := range devices {
		dev := filepath.Join(root, "bus", "pci", "devices", addr)
		for name, content := range files {
			write(t, filepath.Join(dev, name), content)
		}
		if err := os.Symlink("../../../devices/"+addr, filepath.Join(drv, addr)); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func vram(totalMB, usedMB int) map[string]string {
	return map[string]string{
		"mem_info_vram_total": fmt.Sprintf("%d\n", int64(totalMB)<<20),
		"mem_info_vram_used":  fmt.Sprintf("%d\n", int64(usedMB)<<20),
		"vendor":              "0x1002\n",
	}
}

func with(files map[string]string, kv ...string) map[string]string {
	for i := 0; i+1 < len(kv); i += 2 {
		files[kv[i]] = kv[i+1]
	}
	return files
}

func twoGPUs(t *testing.T) string {
	return fakeSys(t, map[string]map[string]string{
		igpu:  with(vram(512, 300), "device", "0x164e\n", "gpu_busy_percent", "3\n"),
		w7800: with(vram(32752, 12000), "device", "0x7448\n", "gpu_busy_percent", "37\n", "power/runtime_status", "active\n"),
	})
}

// fakeProc builds /proc/<pid>/fd and fdinfo: fds maps an fd number to
// "link target|fdinfo content" (no fdinfo when there is no "|").
func fakeProc(t *testing.T, root string, pid int, fds map[int]string) {
	t.Helper()
	dir := filepath.Join(root, fmt.Sprint(pid))
	if err := os.MkdirAll(filepath.Join(dir, "fd"), 0o755); err != nil {
		t.Fatal(err)
	}
	for fd, spec := range fds {
		target, info, hasInfo := strings.Cut(spec, "|")
		if err := os.Symlink(target, filepath.Join(dir, "fd", fmt.Sprint(fd))); err != nil {
			t.Fatal(err)
		}
		if hasInfo {
			write(t, filepath.Join(dir, "fdinfo", fmt.Sprint(fd)), info)
		}
	}
}

func drmInfo(pdev string, client int, vramKey, size string) string {
	return fmt.Sprintf("pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\ndrm-client-id:\t%d\ndrm-pdev:\t%s\n%s:\t%s\n", client, pdev, vramKey, size)
}

func TestDevicesPCIOrder(t *testing.T) {
	sys := fakeSys(t, map[string]map[string]string{
		"0000:7c:00.0":  vram(512, 0),
		"0000:03:00.0":  vram(1, 0),
		"0000:0a:00.0":  vram(1, 0),
		"10000:01:00.0": vram(1, 0), // Intel VMD domain, sorts last
		"0001:01:00.0":  vram(1, 0),
	})
	got, err := Devices(sys)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"0000:03:00.0", "0000:0a:00.0", "0000:7c:00.0", "0001:01:00.0", "10000:01:00.0"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Devices = %v, want %v", got, want)
	}
}

func TestDevicesNoDriver(t *testing.T) {
	got, err := Devices(t.TempDir())
	if err != nil || len(got) != 0 {
		t.Errorf("no amdgpu driver: %v %v, want no devices and no error", got, err)
	}
	if _, err := New(Options{SysfsRoot: t.TempDir()}); err == nil {
		t.Error("New without any AMD GPU must fail")
	}
}

func TestNewIndex(t *testing.T) {
	sys := twoGPUs(t)
	for idx, addr := range []string{w7800, igpu} {
		m, err := New(Options{Index: idx, SysfsRoot: sys, ProcRoot: t.TempDir()})
		if err != nil {
			t.Fatal(err)
		}
		if m.PCIAddr != addr {
			t.Errorf("amdgpu:%d = %s, want %s", idx, m.PCIAddr, addr)
		}
	}
	for _, idx := range []int{2, -1} {
		if _, err := New(Options{Index: idx, SysfsRoot: sys}); err == nil || !strings.Contains(err.Error(), "found 2") {
			t.Errorf("amdgpu:%d: err = %v, want no such GPU, found 2", idx, err)
		}
	}
}

func TestNewUnreadableVRAM(t *testing.T) {
	sys := fakeSys(t, map[string]map[string]string{w7800: {"vendor": "0x1002\n"}})
	if _, err := New(Options{SysfsRoot: sys}); err == nil || !strings.Contains(err.Error(), "mem_info_vram_total") {
		t.Errorf("err = %v, want one naming mem_info_vram_total", err)
	}
}

func TestSnapshot(t *testing.T) {
	usePCIIDs(t, filepath.Join("testdata", "pci.ids"))
	m, err := New(Options{Index: 0, SysfsRoot: twoGPUs(t), ProcRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if s.Name != "AMD Radeon Pro W7800" || s.MemoryKind != gpu.Dedicated || s.TotalMB != 32752 || s.UsedMB != 12000 || s.UtilPct != 37 {
		t.Errorf("snapshot = %+v", s)
	}
	if s.At.IsZero() || len(s.Processes) != 0 {
		t.Errorf("at %v, processes %v", s.At, s.Processes)
	}
}

// An idle card in runtime PM refuses gpu_busy_percent (EBUSY) on current
// kernels and is woken by it on older ones: not asked, -1, no error.
func TestSnapshotUtilUnavailable(t *testing.T) {
	cases := map[string]map[string]string{
		"suspended":   with(vram(32752, 100), "gpu_busy_percent", "0\n", "power/runtime_status", "suspended\n"),
		"no file":     vram(32752, 100),
		"read fails":  with(vram(32752, 100), "gpu_busy_percent/x", ""), // a directory: EISDIR
		"not numeric": with(vram(32752, 100), "gpu_busy_percent", "busy\n"),
	}
	for name, files := range cases {
		m, err := New(Options{SysfsRoot: fakeSys(t, map[string]map[string]string{w7800: files}), ProcRoot: t.TempDir()})
		if err != nil {
			t.Fatal(err)
		}
		s, err := m.Snapshot(context.Background())
		if err != nil || s.UtilPct != -1 || s.TotalMB != 32752 {
			t.Errorf("%s: util %d, total %d, err %v; want -1, 32752, nil", name, s.UtilPct, s.TotalMB, err)
		}
	}
}

func TestSnapshotVRAMError(t *testing.T) {
	sys := twoGPUs(t)
	m, err := New(Options{SysfsRoot: sys, ProcRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	os.Remove(filepath.Join(sys, "bus", "pci", "devices", w7800, "mem_info_vram_used"))
	if _, err := m.Snapshot(context.Background()); err == nil {
		t.Error("unreadable mem_info_vram_used must fail the snapshot")
	}
}

func TestDeviceName(t *testing.T) {
	usePCIIDs(t, filepath.Join(t.TempDir(), "missing"), filepath.Join("testdata", "pci.ids"))
	cases := []struct {
		files map[string]string
		want  string
	}{
		{map[string]string{"product_name": "AMD Radeon PRO W7800\n", "vendor": "0x1002\n", "device": "0x7448\n"}, "AMD Radeon PRO W7800"},
		{map[string]string{"product_name": "\n", "vendor": "0x1002\n", "device": "0x7448\n"}, "AMD Radeon Pro W7800"},
		{map[string]string{"vendor": "0x1002\n", "device": "0x164e\n"}, "AMD Raphael"},
		{map[string]string{"vendor": "0x1002\n", "device": "0x744c\n"}, "AMD Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M"},
		{map[string]string{"vendor": "0x1002\n", "device": "0x0001\n"}, "AMD GPU 1002:0001"},
	}
	for _, c := range cases {
		dir := t.TempDir()
		for name, content := range c.files {
			write(t, filepath.Join(dir, name), content)
		}
		if got := deviceName(dir); got != c.want {
			t.Errorf("deviceName(%v) = %q, want %q", c.files, got, c.want)
		}
	}
}

func TestParseFDInfo(t *testing.T) {
	// Modelled on amdgpu 6.12 (common DRM memory stats plus the legacy
	// keys) and 6.8 (legacy keys only).
	c := ParseFDInfo(fixture(t, "fdinfo-total-vram.txt"))
	if c.Driver != "amdgpu" || c.PDev != w7800 || c.ID != "27" || !c.HasVRAM || c.VRAMBytes != 9216<<20 {
		t.Errorf("total-vram fixture: %+v", c)
	}
	c = ParseFDInfo(fixture(t, "fdinfo-memory-vram.txt"))
	if c.Driver != "amdgpu" || c.PDev != w7800 || c.ID != "14" || !c.HasVRAM || c.VRAMBytes != 2<<30 {
		t.Errorf("memory-vram fixture: %+v", c)
	}
	// drm-total-vram wins over drm-memory-vram, wherever it is.
	c = ParseFDInfo([]byte("drm-memory-vram:\t1024 KiB\ndrm-total-vram:\t4096\n"))
	if c.VRAMBytes != 4096 {
		t.Errorf("drm-total-vram must win: %+v", c)
	}
	// Not a DRM file (/dev/kfd, a socket).
	if c := ParseFDInfo([]byte("pos:\t0\nflags:\t02\nmnt_id:\t15\nino:\t1\n")); c.Driver != "" || c.HasVRAM {
		t.Errorf("non-DRM fdinfo: %+v", c)
	}
}

func TestParseSize(t *testing.T) {
	cases := map[string]int64{
		"0":         0,
		"4095":      4095,
		"512 KiB":   512 << 10,
		"4120 kB":   4120 << 10,
		"9216 MiB":  9216 << 20,
		"2 GiB":     2 << 30,
		" 7 KiB  ":  7 << 10,
		"123 bytes": -1,
		"x KiB":     -1,
		"":          -1,
	}
	for in, want := range cases {
		got, err := parseSize(in)
		if want < 0 {
			if err == nil {
				t.Errorf("parseSize(%q) = %d, want error", in, got)
			}
			continue
		}
		if err != nil || got != want {
			t.Errorf("parseSize(%q) = %d %v, want %d", in, got, err, want)
		}
	}
}

func TestProcesses(t *testing.T) {
	proc := t.TempDir()
	// llama-server built for ROCm: KFD, a render node client on the W7800
	// held through two fds (libdrm dup's the fd it is given), and a client
	// on the iGPU, which ROCm opens too.
	fakeProc(t, proc, 1000, map[int]string{
		0: "/dev/null|pos:\t0\n",
		3: "/dev/kfd|pos:\t0\nflags:\t02\n",
		4: "/dev/dri/renderD128|" + drmInfo(w7800, 10, "drm-total-vram", "8192 MiB"),
		5: "/dev/dri/renderD128|" + drmInfo(w7800, 10, "drm-total-vram", "8192 MiB"),
		6: "/dev/dri/renderD129|" + drmInfo(igpu, 11, "drm-total-vram", "512 MiB"),
		7: "socket:[12345]|pos:\t0\n",
	})
	// llama-server built for Vulkan on an older kernel: two clients on the
	// W7800 (they add up), one on the iGPU.
	fakeProc(t, proc, 2000, map[int]string{
		7: "/dev/dri/renderD128|" + drmInfo(w7800, 20, "drm-memory-vram", "2097152 KiB"),
		8: "/dev/dri/renderD128|" + drmInfo(w7800, 21, "drm-memory-vram", "1048576 KiB"),
		9: "/dev/dri/renderD129|" + drmInfo(igpu, 22, "drm-memory-vram", "65536 KiB"),
	})
	// The desktop, on the iGPU only.
	fakeProc(t, proc, 3000, map[int]string{
		12: "/dev/dri/card1|" + drmInfo(igpu, 30, "drm-total-vram", "180 MiB"),
	})
	// A client on the W7800 that holds no VRAM yet.
	fakeProc(t, proc, 4000, map[int]string{
		3: "/dev/dri/renderD128|" + drmInfo(w7800, 40, "drm-total-vram", "0"),
	})
	// An fd closed between listing and reading its fdinfo.
	fakeProc(t, proc, 5000, map[int]string{3: "/dev/dri/renderD128"})
	// Another driver's device (an NVIDIA card next to the AMD ones).
	fakeProc(t, proc, 6000, map[int]string{
		3: "/dev/dri/renderD130|drm-driver:\tnvidia\ndrm-pdev:\t" + w7800 + "\ndrm-total-vram:\t1 GiB\n",
	})
	// Another user's process: its fd directory cannot be read.
	write(t, filepath.Join(proc, "7000", "fd"), "")
	// Not processes.
	write(t, filepath.Join(proc, "meminfo"), "MemTotal: 1 kB\n")
	if err := os.Symlink("1000", filepath.Join(proc, "self")); err != nil {
		t.Fatal(err)
	}

	sys := twoGPUs(t)
	want := map[int][]gpu.ProcessUsage{
		0: {{PID: 1000, UsedMB: 8192}, {PID: 2000, UsedMB: 3072}, {PID: 4000, UsedMB: 0}},
		1: {{PID: 1000, UsedMB: 512}, {PID: 2000, UsedMB: 64}, {PID: 3000, UsedMB: 180}},
	}
	for idx, w := range want {
		m, err := New(Options{Index: idx, SysfsRoot: sys, ProcRoot: proc})
		if err != nil {
			t.Fatal(err)
		}
		s, err := m.Snapshot(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(s.Processes, w) {
			t.Errorf("amdgpu:%d processes = %v, want %v", idx, s.Processes, w)
		}
	}
}

func TestProcessesNoProc(t *testing.T) {
	m, err := New(Options{SysfsRoot: twoGPUs(t), ProcRoot: filepath.Join(t.TempDir(), "missing")})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Snapshot(context.Background()); err == nil {
		t.Error("an unreadable proc root must fail the snapshot")
	}
}

func TestProcessesCancelled(t *testing.T) {
	proc := t.TempDir()
	fakeProc(t, proc, 1000, map[int]string{4: "/dev/dri/renderD128|" + drmInfo(w7800, 10, "drm-total-vram", "1 MiB")})
	m, err := New(Options{SysfsRoot: twoGPUs(t), ProcRoot: proc})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Snapshot(ctx); err == nil {
		t.Error("a cancelled snapshot must fail")
	}
}
