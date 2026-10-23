// Package amdgpu implements gpu.Monitor for AMD GPUs driven by the amdgpu
// kernel driver, reading the kernel's own interfaces instead of shelling
// out to rocm-smi: sysfs for the device (VRAM total and used, utilisation)
// and the DRM client usage stats in /proc/<pid>/fdinfo for per-process
// VRAM. Both ROCm (HIP) and Vulkan allocate through the render node's GPU
// VM (ROCm's KFD acquires it), so one source covers both llama.cpp builds.
//
// Devices are numbered like rocm-smi does: amdgpu:N is the N-th device
// bound to the amdgpu driver in PCI address order. The N in
// /sys/class/drm/cardN is not used; the kernel does not keep it stable.
package amdgpu

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/w512/gridcore/internal/gpu"
)

const mib = 1 << 20

// Options selects the device and, for tests, the filesystem roots.
type Options struct {
	Index     int    // N from gpu.device "amdgpu:N"
	SysfsRoot string // "" means "/sys"
	ProcRoot  string // "" means "/proc"
}

// Monitor reads one AMD GPU.
type Monitor struct {
	Index   int
	PCIAddr string // e.g. 0000:03:00.0, as in fdinfo's drm-pdev

	name string
	dir  string // the PCI device in sysfs
	proc string
}

// New finds the Index-th AMD GPU and checks that its VRAM can be read.
func New(opts Options) (*Monitor, error) {
	sys, proc := opts.SysfsRoot, opts.ProcRoot
	if sys == "" {
		sys = "/sys"
	}
	if proc == "" {
		proc = "/proc"
	}
	addrs, err := Devices(sys)
	if err != nil {
		return nil, err
	}
	if opts.Index < 0 || opts.Index >= len(addrs) {
		return nil, fmt.Errorf("amdgpu:%d: no such AMD GPU (found %d bound to the amdgpu driver)", opts.Index, len(addrs))
	}
	m := &Monitor{
		Index:   opts.Index,
		PCIAddr: addrs[opts.Index],
		dir:     filepath.Join(sys, "bus", "pci", "devices", addrs[opts.Index]),
		proc:    proc,
	}
	if _, err := readInt(m.dir, "mem_info_vram_total"); err != nil {
		return nil, fmt.Errorf("amdgpu:%d (%s): %w", m.Index, m.PCIAddr, err)
	}
	m.name = deviceName(m.dir)
	return m, nil
}

var pciAddr = regexp.MustCompile(`^([0-9a-f]{4,}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])$`)

// Devices returns the PCI addresses of the devices bound to the amdgpu
// driver, in PCI address order: index N in the result is amdgpu:N. An
// empty sysfsRoot means "/sys". No amdgpu driver loaded means no devices.
func Devices(sysfsRoot string) ([]string, error) {
	if sysfsRoot == "" {
		sysfsRoot = "/sys"
	}
	ents, err := os.ReadDir(filepath.Join(sysfsRoot, "bus", "pci", "drivers", "amdgpu"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("amdgpu: list devices: %w", err)
	}
	var addrs []string
	for _, e := range ents {
		// The driver directory also holds bind, unbind, module, new_id...
		if pciAddr.MatchString(e.Name()) {
			addrs = append(addrs, e.Name())
		}
	}
	sort.Slice(addrs, func(i, j int) bool { return pciLess(addrs[i], addrs[j]) })
	return addrs, nil
}

// pciLess orders domain:bus:device.function numerically; domains can be
// wider than four digits (Intel VMD), so string order is not enough.
func pciLess(a, b string) bool {
	pa, pb := pciAddr.FindStringSubmatch(a), pciAddr.FindStringSubmatch(b)
	for i := 1; i <= 4; i++ {
		x, _ := strconv.ParseUint(pa[i], 16, 64)
		y, _ := strconv.ParseUint(pb[i], 16, 64)
		if x != y {
			return x < y
		}
	}
	return false
}

// Snapshot implements gpu.Monitor.
func (m *Monitor) Snapshot(ctx context.Context) (gpu.Snapshot, error) {
	total, err := readInt(m.dir, "mem_info_vram_total")
	if err != nil {
		return gpu.Snapshot{}, fmt.Errorf("amdgpu:%d: %w", m.Index, err)
	}
	used, err := readInt(m.dir, "mem_info_vram_used")
	if err != nil {
		return gpu.Snapshot{}, fmt.Errorf("amdgpu:%d: %w", m.Index, err)
	}
	procs, err := m.processes(ctx)
	if err != nil {
		return gpu.Snapshot{}, fmt.Errorf("amdgpu:%d: %w", m.Index, err)
	}
	return gpu.Snapshot{
		Name:       m.name,
		MemoryKind: gpu.Dedicated,
		TotalMB:    int(total / mib),
		UsedMB:     int(used / mib),
		UtilPct:    m.util(),
		Processes:  procs,
		At:         time.Now(),
	}, nil
}

// util reads gpu_busy_percent, or -1. A runtime-suspended card is not
// asked at all: current kernels refuse the read (EBUSY), older ones wake
// the card up to answer it, every poll, and so keep it from sleeping.
func (m *Monitor) util() int {
	if s, err := readString(m.dir, filepath.Join("power", "runtime_status")); err == nil && s == "suspended" {
		return -1
	}
	v, err := readInt(m.dir, "gpu_busy_percent")
	if err != nil || v < 0 || v > 100 {
		return -1
	}
	return int(v)
}

func readString(dir, name string) (string, error) {
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(b)), nil
}

func readInt(dir, name string) (int64, error) {
	s, err := readString(dir, name)
	if err != nil {
		return 0, err
	}
	v, err := strconv.ParseInt(s, 0, 64)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", name, err)
	}
	return v, nil
}
