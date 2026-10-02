// Package gpu abstracts "how much memory does the GPU have, who is using it".
//
// The scheduler polls Monitor.Snapshot at gpu.poll_interval and attributes
// per-process usage to model instances by PID. Implementations: nvidia
// (nvidia-smi), apple (Apple Silicon unified memory: ioreg, footprint,
// sysctl), simulation (in-memory, driven by the simulation runtime), and
// later amdgpu (sysfs).
package gpu

import (
	"context"
	"time"
)

// MemoryKind distinguishes discrete VRAM from memory shared with the host.
type MemoryKind string

const (
	Dedicated MemoryKind = "dedicated"
	Unified   MemoryKind = "unified"
)

// Pressure is the host's memory pressure level. On a unified-memory device
// the GPU shares RAM with the OS and every other app, so the scheduler
// watches it alongside the device budget. The values are the macOS kernel's
// (kern.memorystatus_vm_pressure_level); zero means not reported, which is
// always the case on dedicated devices.
type Pressure int

const (
	PressureUnknown  Pressure = 0
	PressureNormal   Pressure = 1
	PressureWarn     Pressure = 2
	PressureCritical Pressure = 4
)

func (p Pressure) String() string {
	switch p {
	case PressureNormal:
		return "normal"
	case PressureWarn:
		return "warn"
	case PressureCritical:
		return "critical"
	}
	return "unknown"
}

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

	// Host memory on unified devices; zero on dedicated ones.
	Pressure   Pressure
	SwapUsedMB int
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

// PIDWatcher is implemented by monitors that cannot list every process on
// the device and measure only the processes they are told about (Apple
// Silicon: per-process memory comes from footprint, PID by PID). The
// scheduler calls Watch with the PIDs of its instances before each poll.
type PIDWatcher interface {
	Watch(pids []int)
}
