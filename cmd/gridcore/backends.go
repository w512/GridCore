package main

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/gridcore/gridcore/internal/config"
	"github.com/gridcore/gridcore/internal/gpu"
	gpufake "github.com/gridcore/gridcore/internal/gpu/fake"
	"github.com/gridcore/gridcore/internal/gpu/nvidia"
	"github.com/gridcore/gridcore/internal/runtime"
	rtfake "github.com/gridcore/gridcore/internal/runtime/fake"
)

// buildBackends constructs the GPU monitor and runtime adapters from config.
func buildBackends(cfg *config.Config) (gpu.Monitor, map[string]runtime.Runtime, error) {
	var mon gpu.Monitor
	var fakeMon *gpufake.Monitor

	device := cfg.GPU.Device
	if device == "auto" {
		if _, err := exec.LookPath("nvidia-smi"); err == nil {
			device = "nvidia:0"
		} else {
			return nil, nil, fmt.Errorf("gpu.device auto: no supported GPU tooling found (nvidia-smi); set gpu.device explicitly (nvidia:N or fake)")
		}
	}
	switch {
	case device == "fake":
		fakeMon = gpufake.New("FakeGPU", cfg.GPU.FakeTotalMB)
		mon = fakeMon
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
		case config.RuntimeFake:
			rt := rtfake.New(name, fakeMon)
			rt.RequestDelay = rc.FakeRequestDelay
			rt.LoadTimeout = rc.LoadTimeout
			runtimes[name] = rt
		case config.RuntimeLlamaCpp:
			return nil, nil, fmt.Errorf("runtimes.%s: llamacpp adapter is not implemented yet (milestone M2)", name)
		default:
			return nil, nil, fmt.Errorf("runtimes.%s: unknown type %q", name, rc.Type)
		}
	}
	return mon, runtimes, nil
}
