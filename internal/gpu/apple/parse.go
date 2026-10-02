package apple

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/w512/gridcore/internal/gpu"
)

const mib = 1024 * 1024

// Device is what `ioreg -r -d 1 -w 0 -c IOAccelerator` says about the GPU.
type Device struct {
	Name    string // "Apple M4 Pro"
	InUseMB int    // GPU memory in use by every process
	UtilPct int    // -1 if not reported
}

var (
	ioregModel = regexp.MustCompile(`"model" = "([^"]+)"`)
	ioregInUse = regexp.MustCompile(`"In use system memory"=(\d+)`)
	ioregUtil  = regexp.MustCompile(`"Device Utilization %"=(\d+)`)
)

// ParseIOReg reads the first accelerator. "In use system memory" is the
// only required key: without it there is no device total to account
// against, and a macOS update that renames it must fail loudly.
func ParseIOReg(out []byte) (Device, error) {
	s := string(out)
	m := ioregInUse.FindStringSubmatch(s)
	if m == nil {
		return Device{}, fmt.Errorf("ioreg: no \"In use system memory\" in IOAccelerator output")
	}
	inUse, err := strconv.ParseInt(m[1], 10, 64)
	if err != nil {
		return Device{}, fmt.Errorf("ioreg: in use system memory: %w", err)
	}
	d := Device{Name: "Apple GPU", InUseMB: int(inUse / mib), UtilPct: -1}
	if m := ioregModel.FindStringSubmatch(s); m != nil {
		d.Name = m[1]
	}
	if m := ioregUtil.FindStringSubmatch(s); m != nil {
		d.UtilPct, _ = strconv.Atoi(m[1])
	}
	return d, nil
}

// "  MTL0: Apple M4 Pro (16384 MiB, 16383 MiB free)"
var listDevice = regexp.MustCompile(`(?m)^\s*MTL\d+:\s*(.+?)\s*\((\d+) MiB,`)

// ParseListDevices returns the first Metal device in `llama-server
// --list-devices` output. Its size is Metal's recommendedMaxWorkingSetSize,
// the most the GPU may keep wired.
func ParseListDevices(out []byte) (name string, totalMB int, err error) {
	m := listDevice.FindStringSubmatch(string(out))
	if m == nil {
		return "", 0, fmt.Errorf("llama-server --list-devices: no Metal device listed")
	}
	totalMB, err = strconv.Atoi(m[2])
	return m[1], totalMB, err
}

var swapUsed = regexp.MustCompile(`used = ([\d.]+)([KMG])`)

// ParseSysctl reads `sysctl -n kern.memorystatus_vm_pressure_level
// vm.swapusage`: the pressure level on the first line, then
// "total = 4096.00M  used = 2452.94M  free = 1643.06M  (encrypted)".
func ParseSysctl(out []byte) (gpu.Pressure, int, error) {
	lines := strings.SplitN(strings.TrimSpace(string(out)), "\n", 2)
	level, err := strconv.Atoi(strings.TrimSpace(lines[0]))
	if err != nil {
		return 0, 0, fmt.Errorf("sysctl: pressure level %q: %w", lines[0], err)
	}
	p := gpu.Pressure(level)
	switch p {
	case gpu.PressureNormal, gpu.PressureWarn, gpu.PressureCritical:
	default:
		p = gpu.PressureUnknown
	}
	swap := 0
	if len(lines) == 2 {
		if m := swapUsed.FindStringSubmatch(lines[1]); m != nil {
			v, _ := strconv.ParseFloat(m[1], 64)
			switch m[2] {
			case "K":
				v /= 1024
			case "G":
				v *= 1024
			}
			swap = int(v)
		}
	}
	return p, swap, nil
}

type footprintJSON struct {
	Processes []struct {
		PID        int                       `json:"pid"`
		Footprint  int64                     `json:"footprint"`
		Categories map[string]footprintBytes `json:"categories"`
	} `json:"processes"`
}

type footprintBytes struct {
	Dirty int64 `json:"dirty"`
	Clean int64 `json:"clean"`
}

// ParseFootprint reads `footprint -f bytes -j FILE pid...`. A process's
// usage is its phys_footprint plus the clean resident pages of mapped
// files: with mmap the model weights live there, outside phys_footprint,
// and still take RAM. On a unified device CPU-side and GPU-side memory are
// the same pool, so this whole number is what the model costs.
func ParseFootprint(raw []byte) ([]gpu.ProcessUsage, error) {
	var d footprintJSON
	if err := json.Unmarshal(raw, &d); err != nil {
		return nil, fmt.Errorf("footprint: %w", err)
	}
	procs := make([]gpu.ProcessUsage, 0, len(d.Processes))
	for _, p := range d.Processes {
		b := p.Footprint + p.Categories["mapped file"].Clean
		procs = append(procs, gpu.ProcessUsage{PID: p.PID, UsedMB: int(b / mib)})
	}
	return procs, nil
}

// EstimateLimitMB is Metal's default working-set limit when nothing better
// is available: two thirds of RAM (measured: 16384 MiB on a 24 GB M4 Pro),
// three quarters above 36 GB as macOS is reported to allow there.
func EstimateLimitMB(memBytes int64) int {
	ram := memBytes / mib
	if ram <= 36*1024 {
		return int(ram * 2 / 3)
	}
	return int(ram * 3 / 4)
}
