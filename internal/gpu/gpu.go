// Package gpu abstracts "how much memory does the GPU have, who is using it".
//
// The scheduler polls Monitor.Snapshot at gpu.poll_interval and attributes
// per-process usage to model instances by PID. Implementations: nvidia
// (nvidia-smi), fake (in-memory, driven by the fake runtime), and later
// amdgpu (sysfs) and unified-memory devices.
package gpu

import (
	"context"
	"time"
)

// MemoryKind distinguishes discrete VRAM from memory shared with the host.
// v0.1 only handles Dedicated; the field exists so the planner can branch
// later without changing the interface.
type MemoryKind string

const (
	Dedicated MemoryKind = "dedicated"
	Unified   MemoryKind = "unified"
)

// ProcessUsage is one process's footprint on the device.
type ProcessUsage struct {
	PID    int
	UsedMB int
}

// Snapshot is a point-in-time reading of one device.
type Snapshot struct {
	Name       string
	MemoryKind MemoryKind
	TotalMB    int
	UsedMB     int // all users, including non-GridCore processes
	UtilPct    int // 0-100; -1 if unavailable
	Processes  []ProcessUsage
	At         time.Time
}

// FreeMB is memory not currently used by anyone.
func (s Snapshot) FreeMB() int {
	if f := s.TotalMB - s.UsedMB; f > 0 {
		return f
	}
	return 0
}

// UsedByPID returns the usage attributed to pid, or 0.
func (s Snapshot) UsedByPID(pid int) int {
	for _, p := range s.Processes {
		if p.PID == pid {
			return p.UsedMB
		}
	}
	return 0
}

// Monitor reads device state.
type Monitor interface {
	// Snapshot returns the current state. Implementations should be cheap
	// enough to call every few hundred milliseconds.
	Snapshot(ctx context.Context) (Snapshot, error)
}
