// Package nvidia implements gpu.Monitor on top of nvidia-smi.
//
// It shells out rather than binding NVML so the binary stays cgo-free and
// cross-compiles from macOS. Two queries per snapshot, ~20-40 ms total,
// which is fine at a 500 ms poll interval. NVML can replace the exec path
// later without touching the parser or the interface.
package nvidia

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/w512/gridcore/internal/gpu"
)

const (
	gpuQuery  = "name,memory.total,memory.used,utilization.gpu"
	procQuery = "pid,used_memory"
	csvFormat = "csv,noheader,nounits"
)

// Monitor reads one device by index.
type Monitor struct {
	Index  int
	Binary string // defaults to "nvidia-smi" on PATH
}

// New creates a monitor for device idx.
func New(idx int) *Monitor { return &Monitor{Index: idx, Binary: "nvidia-smi"} }

// Snapshot implements gpu.Monitor.
func (m *Monitor) Snapshot(ctx context.Context) (gpu.Snapshot, error) {
	id := strconv.Itoa(m.Index)
	gpuOut, err := m.run(ctx, "--id="+id, "--query-gpu="+gpuQuery, "--format="+csvFormat)
	if err != nil {
		return gpu.Snapshot{}, err
	}
	procOut, err := m.run(ctx, "--id="+id, "--query-compute-apps="+procQuery, "--format="+csvFormat)
	if err != nil {
		return gpu.Snapshot{}, err
	}
	snap, err := ParseGPU(gpuOut)
	if err != nil {
		return gpu.Snapshot{}, err
	}
	procs, err := ParseProcesses(procOut)
	if err != nil {
		return gpu.Snapshot{}, err
	}
	snap.Processes = procs
	snap.At = time.Now()
	return snap, nil
}

func (m *Monitor) run(ctx context.Context, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, m.Binary, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return out, nil
}

// ParseGPU parses one line of `--query-gpu=name,memory.total,memory.used,
// utilization.gpu --format=csv,noheader,nounits`.
func ParseGPU(out []byte) (gpu.Snapshot, error) {
	line := strings.TrimSpace(string(out))
	if line == "" {
		return gpu.Snapshot{}, fmt.Errorf("nvidia-smi: empty gpu query output")
	}
	// Only the first line matters when --id is used; tolerate extra lines.
	if i := strings.IndexByte(line, '\n'); i >= 0 {
		line = line[:i]
	}
	f := splitCSV(line)
	if len(f) != 4 {
		return gpu.Snapshot{}, fmt.Errorf("nvidia-smi: want 4 gpu fields, got %d in %q", len(f), line)
	}
	total, err := atoi(f[1])
	if err != nil {
		return gpu.Snapshot{}, fmt.Errorf("nvidia-smi: memory.total: %w", err)
	}
	used, err := atoi(f[2])
	if err != nil {
		return gpu.Snapshot{}, fmt.Errorf("nvidia-smi: memory.used: %w", err)
	}
	util := -1
	if v, err := atoi(f[3]); err == nil {
		util = v
	}
	return gpu.Snapshot{
		Name:       f[0],
		MemoryKind: gpu.Dedicated,
		TotalMB:    total,
		UsedMB:     used,
		UtilPct:    util,
	}, nil
}

// ParseProcesses parses `--query-compute-apps=pid,used_memory` output.
// Empty output (no compute processes) is valid.
func ParseProcesses(out []byte) ([]gpu.ProcessUsage, error) {
	var procs []gpu.ProcessUsage
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		f := splitCSV(line)
		if len(f) != 2 {
			return nil, fmt.Errorf("nvidia-smi: want 2 process fields, got %d in %q", len(f), line)
		}
		pid, err := atoi(f[0])
		if err != nil {
			return nil, fmt.Errorf("nvidia-smi: pid: %w", err)
		}
		// used_memory is "[N/A]" on some drivers/WSL; attribute 0 rather than fail.
		mb, err := atoi(f[1])
		if err != nil {
			mb = 0
		}
		procs = append(procs, gpu.ProcessUsage{PID: pid, UsedMB: mb})
	}
	return procs, nil
}

func splitCSV(line string) []string {
	parts := strings.Split(line, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}

func atoi(s string) (int, error) {
	return strconv.Atoi(strings.TrimSpace(s))
}
