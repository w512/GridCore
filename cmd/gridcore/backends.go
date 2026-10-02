package main

import (
	"fmt"
	"log/slog"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/w512/gridcore/internal/config"
	"github.com/w512/gridcore/internal/gpu"
	"github.com/w512/gridcore/internal/gpu/nvidia"
	gpusim "github.com/w512/gridcore/internal/gpu/simulation"
	"github.com/w512/gridcore/internal/runtime"
	"github.com/w512/gridcore/internal/runtime/llamacpp"
	rtsim "github.com/w512/gridcore/internal/runtime/simulation"
)

// buildBackends constructs the GPU monitor and runtime adapters from config.
// When stateDir is non-empty, llama.cpp children are logged there and
// recorded for orphan cleanup on the next start.
func buildBackends(cfg *config.Config, stateDir string) (gpu.Monitor, map[string]runtime.Runtime, error) {
	var mon gpu.Monitor
	var simMon *gpusim.Monitor

	device := cfg.GPU.Device
	if device == "auto" {
		if _, err := exec.LookPath("nvidia-smi"); err == nil {
			device = "nvidia:0"
		} else {
			return nil, nil, fmt.Errorf("gpu.device auto: no supported GPU tooling found (nvidia-smi); set gpu.device explicitly (nvidia:N or simulation)")
		}
	}
	switch {
	case device == "simulation":
		simMon = gpusim.New("Simulated GPU", cfg.GPU.SimulatedTotalMB)
		mon = simMon
	case strings.HasPrefix(device, "nvidia:"):
		idx, err := strconv.Atoi(strings.TrimPrefix(device, "nvidia:"))
		if err != nil {
			return nil, nil, fmt.Errorf("gpu.device %q: bad index", device)
		}
		mon = nvidia.New(idx)
	default:
		return nil, nil, fmt.Errorf("gpu.device %q is not supported", device)
	}

	runtimes := map[string]runtime.Runtime{}
	for name, rc := range cfg.Runtimes {
		switch rc.Type {
		case config.RuntimeSimulation:
			rt := rtsim.New(name, simMon)
			rt.RequestDelay = rc.SimulatedRequestDelay
			rt.LoadTimeout = rc.LoadTimeout
			runtimes[name] = rt
		case config.RuntimeLlamaCpp:
			var logDir string
			if stateDir != "" {
				logDir = filepath.Join(stateDir, "logs")
			}
			rt := llamacpp.New(name, rc, logDir)
			if stateDir != "" {
				reg, err := llamacpp.OpenRegistry(filepath.Join(stateDir, "instances-"+name+".json"))
				if err != nil {
					return nil, nil, err
				}
				if reaped := reg.ReapOrphans(); len(reaped) > 0 {
					for _, e := range reaped {
						slog.Warn("killed orphaned instance from previous run", "runtime", name, "model", e.Model, "pid", e.PID, "port", e.Port)
					}
				}
				rt.Registry = reg
			}
			runtimes[name] = rt
		default:
			return nil, nil, fmt.Errorf("runtimes.%s: unknown type %q", name, rc.Type)
		}
	}
	return mon, runtimes, nil
}
