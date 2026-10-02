// Package apple implements gpu.Monitor for Apple Silicon, where the GPU
// shares the machine's RAM with the OS and every app.
//
// Like the nvidia monitor it shells out instead of using cgo: ioreg for the
// device (GPU memory in use, utilisation), sysctl for memory pressure and
// swap, footprint for per-process memory. footprint cannot cheaply list
// every process using the GPU, so the scheduler names the PIDs it manages
// (gpu.PIDWatcher). Measured costs on an M4 Pro: ioreg ~14 ms, sysctl ~2 ms,
// footprint ~110 ms for two processes.
package apple

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/w512/gridcore/internal/gpu"
)

// Monitor reads the one GPU of an Apple Silicon machine.
type Monitor struct {
	// TotalMB is Metal's working-set limit, the device's "total" for
	// admission. Source says where it came from.
	TotalMB int
	Source  string

	mu   sync.Mutex
	pids []int
}

// New resolves the Metal working-set limit: from `llama-server
// --list-devices` when a binary is given, else from iogpu.wired_limit_mb
// when it is raised, else estimated from the amount of RAM.
func New(ctx context.Context, llamaServer string) (*Monitor, error) {
	if llamaServer != "" {
		lctx, cancel := context.WithTimeout(ctx, 15*time.Second)
		out, err := exec.CommandContext(lctx, llamaServer, "--list-devices").CombinedOutput()
		cancel()
		if err == nil {
			if _, total, perr := ParseListDevices(out); perr == nil && total > 0 {
				return &Monitor{TotalMB: total, Source: "llama-server --list-devices"}, nil
			}
		}
	}
	out, err := run(ctx, "sysctl", "-n", "iogpu.wired_limit_mb", "hw.memsize")
	if err != nil {
		return nil, err
	}
	f := strings.Fields(string(out))
	if len(f) != 2 {
		return nil, fmt.Errorf("sysctl iogpu.wired_limit_mb hw.memsize: unexpected output %q", out)
	}
	if wired, _ := strconv.Atoi(f[0]); wired > 0 {
		return &Monitor{TotalMB: wired, Source: "iogpu.wired_limit_mb"}, nil
	}
	mem, err := strconv.ParseInt(f[1], 10, 64)
	if err != nil || mem <= 0 {
		return nil, fmt.Errorf("sysctl hw.memsize: %q", f[1])
	}
	return &Monitor{TotalMB: EstimateLimitMB(mem), Source: fmt.Sprintf("estimated from %d GB of RAM", mem>>30)}, nil
}

// Watch implements gpu.PIDWatcher.
func (m *Monitor) Watch(pids []int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pids = append(m.pids[:0], pids...)
}

// Snapshot implements gpu.Monitor.
func (m *Monitor) Snapshot(ctx context.Context) (gpu.Snapshot, error) {
	out, err := run(ctx, "ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator")
	if err != nil {
		return gpu.Snapshot{}, err
	}
	dev, err := ParseIOReg(out)
	if err != nil {
		return gpu.Snapshot{}, err
	}
	out, err = run(ctx, "sysctl", "-n", "kern.memorystatus_vm_pressure_level", "vm.swapusage")
	if err != nil {
		return gpu.Snapshot{}, err
	}
	pressure, swap, err := ParseSysctl(out)
	if err != nil {
		return gpu.Snapshot{}, err
	}
	m.mu.Lock()
	pids := append([]int(nil), m.pids...)
	m.mu.Unlock()
	procs, err := footprint(ctx, pids)
	if err != nil {
		return gpu.Snapshot{}, err
	}
	return gpu.Snapshot{
		Name:       dev.Name,
		MemoryKind: gpu.Unified,
		TotalMB:    m.TotalMB,
		UsedMB:     dev.InUseMB,
		UtilPct:    dev.UtilPct,
		Processes:  procs,
		At:         time.Now(),
		Pressure:   pressure,
		SwapUsedMB: swap,
	}, nil
}

// footprint measures pids in one call. footprint writes JSON only to a
// file, skips PIDs that no longer exist and exits 66 when none of them do.
func footprint(ctx context.Context, pids []int) ([]gpu.ProcessUsage, error) {
	if len(pids) == 0 {
		return nil, nil
	}
	f, err := os.CreateTemp("", "gridcore-footprint-*.json")
	if err != nil {
		return nil, err
	}
	path := f.Name()
	f.Close()
	defer os.Remove(path)
	args := []string{"-f", "bytes", "-j", path}
	for _, p := range pids {
		args = append(args, strconv.Itoa(p))
	}
	cmd := exec.CommandContext(ctx, "footprint", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.ExitCode() == 66 {
			return nil, nil // every watched process is gone
		}
		return nil, fmt.Errorf("footprint: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseFootprint(raw)
}

func run(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("%s %s: %w: %s", name, strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return out, nil
}
