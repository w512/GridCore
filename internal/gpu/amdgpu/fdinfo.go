package amdgpu

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/w512/gridcore/internal/gpu"
)

// Client is one DRM client (an open of a /dev/dri node) as its fdinfo
// describes it.
type Client struct {
	Driver    string // drm-driver
	PDev      string // drm-pdev, the PCI address of the device
	ID        string // drm-client-id
	VRAMBytes int64
	HasVRAM   bool
}

// ParseFDInfo parses /proc/<pid>/fdinfo/<fd> of a DRM file. VRAM is
// drm-total-vram (kernels with the common DRM memory stats) or else the
// amdgpu-specific drm-memory-vram of older kernels. Kernels in between
// print both; the common key wins.
func ParseFDInfo(b []byte) Client {
	var c Client
	var total, legacy string
	for _, line := range strings.Split(string(b), "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		v = strings.TrimSpace(v)
		switch strings.TrimSpace(k) {
		case "drm-driver":
			c.Driver = v
		case "drm-pdev":
			c.PDev = strings.ToLower(v)
		case "drm-client-id":
			c.ID = v
		case "drm-total-vram":
			total = v
		case "drm-memory-vram":
			legacy = v
		}
	}
	for _, s := range []string{total, legacy} {
		if s == "" {
			continue
		}
		if n, err := parseSize(s); err == nil {
			c.VRAMBytes, c.HasVRAM = n, true
			break
		}
	}
	return c
}

// parseSize parses fdinfo sizes: "4096", "512 KiB", "9216 MiB".
func parseSize(s string) (int64, error) {
	num, unit, _ := strings.Cut(strings.TrimSpace(s), " ")
	n, err := strconv.ParseInt(num, 10, 64)
	if err != nil {
		return 0, err
	}
	switch strings.TrimSpace(unit) {
	case "":
		return n, nil
	case "KiB", "kB":
		return n << 10, nil
	case "MiB":
		return n << 20, nil
	case "GiB":
		return n << 30, nil
	}
	return 0, fmt.Errorf("unknown unit in %q", s)
}

// processes sums, per process, the VRAM of its DRM clients on this device.
//
// A process can hold several fds of one client (dup, or libdrm duplicating
// the fd it is given, as ROCm's runtime does), all with the same fdinfo:
// clients are counted once per process by drm-client-id. A process can
// also have clients on other GPUs (Vulkan opens every device, ROCm every
// GPU in its topology, the iGPU included): only drm-pdev of this device
// counts. Processes that exit mid-scan or that we may not inspect (other
// users') are skipped.
func (m *Monitor) processes(ctx context.Context) ([]gpu.ProcessUsage, error) {
	ents, err := os.ReadDir(m.proc)
	if err != nil {
		return nil, fmt.Errorf("list processes: %w", err)
	}
	var procs []gpu.ProcessUsage
	for _, e := range ents {
		pid, err := strconv.Atoi(e.Name())
		if err != nil || pid <= 0 {
			continue
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if bytes, ok := m.processVRAM(filepath.Join(m.proc, e.Name())); ok {
			procs = append(procs, gpu.ProcessUsage{PID: pid, UsedMB: int(bytes / mib)})
		}
	}
	sort.Slice(procs, func(i, j int) bool { return procs[i].PID < procs[j].PID })
	return procs, nil
}

// processVRAM returns the process's VRAM on this device and whether it has
// any client on it at all.
func (m *Monitor) processVRAM(dir string) (int64, bool) {
	fds, err := os.ReadDir(filepath.Join(dir, "fd"))
	if err != nil {
		return 0, false
	}
	seen := map[string]bool{}
	var sum int64
	found := false
	for _, fd := range fds {
		// Read fdinfo only for DRM nodes; a readlink is far cheaper.
		target, err := os.Readlink(filepath.Join(dir, "fd", fd.Name()))
		if err != nil || !strings.HasPrefix(target, "/dev/dri/") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, "fdinfo", fd.Name()))
		if err != nil {
			continue
		}
		c := ParseFDInfo(b)
		if c.Driver != "amdgpu" || c.PDev != m.PCIAddr || !c.HasVRAM {
			continue
		}
		found = true
		if c.ID != "" {
			if seen[c.ID] {
				continue
			}
			seen[c.ID] = true
		}
		sum += c.VRAMBytes
	}
	return sum, found
}
