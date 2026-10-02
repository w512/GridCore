// Package simulation is an in-memory GPU monitor for tests and demos.
//
// The simulation runtime attaches/detaches its instances here so the scheduler
// sees exactly the same per-PID accounting it would get from nvidia-smi.
package simulation

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/w512/gridcore/internal/gpu"
)

// Monitor simulates a single GPU: dedicated memory by default, unified
// after SetUnified.
type Monitor struct {
	name    string
	totalMB int

	mu        sync.Mutex
	procs     map[int]int // pid -> MB
	lingering map[int]int // pid -> MB still counted in memory.used after the process left the list
	external  int         // MB used by "someone else" (desktop, other apps)
	util      int

	// Unified memory: the device then reports host memory pressure and swap.
	unified  bool
	pressure gpu.Pressure
	swapMB   int
}

// New creates a simulated GPU with the given capacity.
func New(name string, totalMB int) *Monitor {
	return &Monitor{name: name, totalMB: totalMB, procs: map[int]int{}, lingering: map[int]int{}}
}

// Attach records pid as using mb of VRAM.
func (m *Monitor) Attach(pid, mb int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.procs[pid] = mb
}

// Detach removes pid and frees its memory at once.
func (m *Monitor) Detach(pid int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.procs, pid)
	delete(m.lingering, pid)
}

// Linger models process teardown as nvidia-smi shows it: pid is gone from
// the per-process list, but its memory is still part of memory.used until
// Release (or Detach) is called.
func (m *Monitor) Linger(pid int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if mb, ok := m.procs[pid]; ok {
		m.lingering[pid] = mb
		delete(m.procs, pid)
	}
}

// Release frees memory left behind by Linger.
func (m *Monitor) Release(pid int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.lingering, pid)
}

// SetExternal sets memory consumed by processes GridCore does not manage.
func (m *Monitor) SetExternal(mb int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.external = mb
}

// SetUnified makes the device report unified memory, with host memory
// pressure (normal unless SetPressure says otherwise) and swap.
func (m *Monitor) SetUnified(on bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.unified = on
}

// SetPressure sets the host memory pressure a unified device reports.
func (m *Monitor) SetPressure(p gpu.Pressure) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pressure = p
}

// SetSwap sets the host swap in use a unified device reports.
func (m *Monitor) SetSwap(mb int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.swapMB = mb
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
	if m.unified {
		s.MemoryKind = gpu.Unified
		s.Pressure = m.pressure
		if s.Pressure == gpu.PressureUnknown {
			s.Pressure = gpu.PressureNormal
		}
		s.SwapUsedMB = m.swapMB
	}
	for pid, mb := range m.procs {
		s.UsedMB += mb
		s.Processes = append(s.Processes, gpu.ProcessUsage{PID: pid, UsedMB: mb})
	}
	for _, mb := range m.lingering {
		s.UsedMB += mb
	}
	sort.Slice(s.Processes, func(i, j int) bool { return s.Processes[i].PID < s.Processes[j].PID })
	if s.UsedMB > s.TotalMB {
		s.UsedMB = s.TotalMB // a real GPU would have OOM'd; clamp for sanity
	}
	return s, nil
}
