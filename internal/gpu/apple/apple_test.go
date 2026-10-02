package apple

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/w512/gridcore/internal/gpu"
)

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// Recorded on an M4 Pro 24 GB, macOS 15.8.1 (IOReportLegend stripped).
func TestParseIOReg(t *testing.T) {
	d, err := ParseIOReg(fixture(t, "ioreg-m4pro.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if d.Name != "Apple M4 Pro" || d.InUseMB != 1278 || d.UtilPct != 15 {
		t.Errorf("got %+v, want Apple M4 Pro, 1278 MB in use, 15%%", d)
	}
}

func TestParseIORegMissingKey(t *testing.T) {
	if _, err := ParseIOReg([]byte(`+-o AGXAcceleratorG16X  <class AGXAcceleratorG16X>` + "\n" + `"model" = "Apple M4 Pro"`)); err == nil {
		t.Error("output without \"In use system memory\" must be an error")
	}
	d, err := ParseIOReg([]byte(`"PerformanceStatistics" = {"In use system memory"=2147483648}`))
	if err != nil || d.Name != "Apple GPU" || d.InUseMB != 2048 || d.UtilPct != -1 {
		t.Errorf("minimal output: %+v, %v", d, err)
	}
}

func TestParseListDevices(t *testing.T) {
	name, total, err := ParseListDevices(fixture(t, "list-devices-m4pro.txt"))
	if err != nil || name != "Apple M4 Pro" || total != 16384 {
		t.Errorf("got %q %d %v, want Apple M4 Pro 16384", name, total, err)
	}
	if _, _, err := ParseListDevices([]byte("Available devices:\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n")); err == nil {
		t.Error("a list without a Metal device must be an error")
	}
}

func TestParseSysctl(t *testing.T) {
	p, swap, err := ParseSysctl(fixture(t, "sysctl-m4pro.txt"))
	if err != nil || p != gpu.PressureNormal || swap != 2452 {
		t.Errorf("got %v %d %v, want normal 2452", p, swap, err)
	}
	cases := []struct {
		in    string
		p     gpu.Pressure
		swap  int
		isErr bool
	}{
		{"2\ntotal = 11264.00M  used = 10146.88M  free = 1117.12M  (encrypted)\n", gpu.PressureWarn, 10146, false},
		{"4\ntotal = 2.00G  used = 1.50G  free = 0.50G\n", gpu.PressureCritical, 1536, false},
		{"7\n", gpu.PressureUnknown, 0, false},
		{"x\n", 0, 0, true},
	}
	for _, c := range cases {
		p, swap, err := ParseSysctl([]byte(c.in))
		if (err != nil) != c.isErr || p != c.p || swap != c.swap {
			t.Errorf("ParseSysctl(%q) = %v %d %v, want %v %d err=%v", c.in, p, swap, err, c.p, c.swap, c.isErr)
		}
	}
}

// A gemma4-e2b llama-server with --load-mode none: 4838 MB phys_footprint
// (weights as anonymous memory) + 23 MB of clean mapped-file pages.
func TestParseFootprint(t *testing.T) {
	procs, err := ParseFootprint(fixture(t, "footprint-llama-server.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(procs) != 1 || procs[0].PID != 3774 || procs[0].UsedMB != 4861 {
		t.Errorf("got %+v, want pid 3774 using 4861 MB", procs)
	}
}

// With mmap the weights are clean file pages outside phys_footprint; they
// still take RAM and must count.
func TestParseFootprintMapped(t *testing.T) {
	raw := `{"processes":[{"pid":7,"footprint":1426063360,"categories":{"mapped file":{"dirty":0,"clean":3306160128},"IOAccelerator (graphics)":{"dirty":90177536,"clean":0}}}]}`
	procs, err := ParseFootprint([]byte(raw))
	if err != nil || len(procs) != 1 || procs[0].UsedMB != 1360+3153 {
		t.Errorf("got %+v %v, want 4513 MB", procs, err)
	}
}

// Runs the real commands; only meaningful on a Mac.
func TestMonitorLive(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("needs macOS")
	}
	m, err := New(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if m.TotalMB <= 0 || m.Source == "" {
		t.Fatalf("limit %d from %q", m.TotalMB, m.Source)
	}
	m.Watch([]int{os.Getpid(), 999999})
	s, err := m.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if s.MemoryKind != gpu.Unified || s.TotalMB != m.TotalMB || s.UsedMB <= 0 || s.Pressure == gpu.PressureUnknown {
		t.Errorf("snapshot %+v", s)
	}
	if len(s.Processes) != 1 || s.Processes[0].PID != os.Getpid() || s.Processes[0].UsedMB <= 0 {
		t.Errorf("processes %+v, want only this test process", s.Processes)
	}
	m.Watch([]int{999999})
	if s, err := m.Snapshot(context.Background()); err != nil || len(s.Processes) != 0 {
		t.Errorf("only dead PIDs: %+v %v", s.Processes, err)
	}
}

func TestEstimateLimitMB(t *testing.T) {
	if got := EstimateLimitMB(24 << 30); got != 16384 {
		t.Errorf("24 GB: %d, want 16384 (measured on an M4 Pro)", got)
	}
	if got := EstimateLimitMB(64 << 30); got != 49152 {
		t.Errorf("64 GB: %d, want 49152", got)
	}
}
