// Package fake is an in-memory GPU monitor for tests and demos.
//
// The fake runtime attaches/detaches its instances here so the scheduler
// sees exactly the same per-PID accounting it would get from nvidia-smi.
package fake

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/gridcore/gridcore/internal/gpu"
)

// Monitor simulates a single dedicated-memory GPU.
type Monitor struct {
	name    string
	totalMB int

	mu       sync.Mutex
	procs    map[int]int // pid -> MB
	external int         // MB used by "someone else" (desktop, other apps)
	util     int
}

// New creates a fake GPU with the given capacity.
func New(name string, totalMB int) *Monitor {
	return &Monitor{name: name, totalMB: totalMB, procs: map[int]int{}}
}

// Attach records pid as using mb of VRAM.
func (m *Monitor) Attach(pid, mb int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.procs[pid] = mb
}

// Detach removes pid.
func (m *Monitor) Detach(pid int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.procs, pid)
}

// SetExternal sets memory consumed by processes GridCore does not manage.
func (m *Monitor) SetExternal(mb int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.external = mb
}

// SetUtil sets the reported utilisation percentage.
func (m *Monitor) SetUtil(pct int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.util = pct
}

// Snapshot implements gpu.Monitor.
func (m *Monitor) Snapshot(_ context.Context) (gpu.Snapshot, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	s := gpu.Snapshot{
		Name:       m.name,
		MemoryKind: gpu.Dedicated,
		TotalMB:    m.totalMB,
		UsedMB:     m.external,
		UtilPct:    m.util,
		At:         time.Now(),
	}
	for pid, mb := range m.procs {
		s.UsedMB += mb
		s.Processes = append(s.Processes, gpu.ProcessUsage{PID: pid, UsedMB: mb})
	}
	sort.Slice(s.Processes, func(i, j int) bool { return s.Processes[i].PID < s.Processes[j].PID })
	if s.UsedMB > s.TotalMB {
		s.UsedMB = s.TotalMB // a real GPU would have OOM'd; clamp for sanity
	}
	return s, nil
}
