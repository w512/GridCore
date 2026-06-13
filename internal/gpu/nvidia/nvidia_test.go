package nvidia

import "testing"

func TestParseGPU(t *testing.T) {
	snap, err := ParseGPU([]byte("NVIDIA GeForce RTX 4060 Ti, 16380, 9215, 37\n"))
	if err != nil {
		t.Fatal(err)
	}
	if snap.Name != "NVIDIA GeForce RTX 4060 Ti" || snap.TotalMB != 16380 || snap.UsedMB != 9215 || snap.UtilPct != 37 {
		t.Errorf("snapshot = %+v", snap)
	}
	if snap.FreeMB() != 16380-9215 {
		t.Errorf("free = %d", snap.FreeMB())
	}
}

func TestParseGPUUtilNA(t *testing.T) {
	snap, err := ParseGPU([]byte("Some GPU, 8192, 100, [N/A]"))
	if err != nil {
		t.Fatal(err)
	}
	if snap.UtilPct != -1 {
		t.Errorf("util should be -1 for N/A, got %d", snap.UtilPct)
	}
}

func TestParseGPUErrors(t *testing.T) {
	for _, in := range []string{"", "a, b", "GPU, x, 1, 2"} {
		if _, err := ParseGPU([]byte(in)); err == nil {
			t.Errorf("expected error for %q", in)
		}
	}
}

func TestParseProcesses(t *testing.T) {
	procs, err := ParseProcesses([]byte("12345, 9004\n12399, 611\n\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(procs) != 2 || procs[0].PID != 12345 || procs[0].UsedMB != 9004 || procs[1].UsedMB != 611 {
		t.Errorf("procs = %+v", procs)
	}
}

func TestParseProcessesEmptyAndNA(t *testing.T) {
	procs, err := ParseProcesses([]byte(""))
	if err != nil || len(procs) != 0 {
		t.Errorf("empty output should give no procs, got %v %v", procs, err)
	}
	procs, err = ParseProcesses([]byte("777, [N/A]"))
	if err != nil || len(procs) != 1 || procs[0].UsedMB != 0 {
		t.Errorf("N/A memory should parse as 0, got %v %v", procs, err)
	}
	if _, err := ParseProcesses([]byte("abc, 1")); err == nil {
		t.Error("bad pid should error")
	}
}
