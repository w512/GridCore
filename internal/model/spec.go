// Package model holds the static description of a model variant (Spec) and
// the measured, persisted facts about it on this machine (Profile).
//
// A "model" for GridCore is a variant with launch args, not a file: VRAM and
// load time depend on (path, ctx, parallel, args, runtime binary, GPU).
package model

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/w512/gridcore/internal/config"
)

// Spec is the scheduler's immutable view of a configured model.
type Spec struct {
	ID          string
	Runtime     string // runtime name in config (e.g. "llamacpp", "llamacpp-hip")
	RuntimeType string // adapter type (config.RuntimeLlamaCpp | config.RuntimeSimulation)

	Path         string
	MMProj       string
	Capabilities []string
	Ctx          int
	Parallel     int
	Args         []string
	Aliases      []string

	Pinned  bool
	Preload bool

	// Simulation knobs.
	SimulatedVRAMMB int
	SimulatedLoad   time.Duration
}

// FromConfig materialises Specs for every configured model.
func FromConfig(cfg *config.Config) map[string]*Spec {
	out := make(map[string]*Spec, len(cfg.Models))
	for id, m := range cfg.Models {
		out[id] = &Spec{
			ID:              id,
			Runtime:         m.Runtime,
			RuntimeType:     cfg.Runtimes[m.Runtime].Type,
			Path:            m.Path,
			MMProj:          m.MMProj,
			Capabilities:    append([]string(nil), m.Capabilities...),
			Ctx:             m.Ctx,
			Parallel:        m.Parallel,
			Args:            append([]string(nil), m.Args...),
			Aliases:         append([]string(nil), m.Aliases...),
			Pinned:          m.Pinned,
			Preload:         m.Preload,
			SimulatedVRAMMB: m.SimulatedVRAMMB,
			SimulatedLoad:   m.SimulatedLoad,
		}
	}
	return out
}

// HasCapability reports whether the model declares cap.
func (s *Spec) HasCapability(cap string) bool {
	for _, c := range s.Capabilities {
		if c == cap {
			return true
		}
	}
	return false
}

// ProfileKey identifies the (variant, runtime build, gpu) tuple a Profile
// belongs to. runtimeID names the runtime build (scheduler.RuntimeID) so
// that a different llama.cpp starts with fresh measurements; gpuName is the
// device name reported by the monitor.
func (s *Spec) ProfileKey(runtimeID, gpuName string) string {
	args := append([]string(nil), s.Args...)
	sort.Strings(args)
	path := s.Path
	if path == "" { // simulated models have no file; identify them by id
		path = "id:" + s.ID
	}
	parts := []string{
		s.RuntimeType, path, s.MMProj,
		fmt.Sprint(s.Ctx), fmt.Sprint(s.Parallel),
		strings.Join(args, "\x00"),
		runtimeID, gpuName,
	}
	sum := sha256.Sum256([]byte(strings.Join(parts, "\x01")))
	return hex.EncodeToString(sum[:8])
}

// EstimateVRAMMB is the pre-measurement admission size. See EstimateVRAM
// for the breakdown.
func (s *Spec) EstimateVRAMMB() int {
	return EstimateVRAM(s).TotalMB
}
