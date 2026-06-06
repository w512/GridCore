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
	"os"
	"sort"
	"strings"
	"time"

	"github.com/gridcore/gridcore/internal/config"
)

// Spec is the scheduler's immutable view of a configured model.
type Spec struct {
	ID          string
	Runtime     string // runtime name in config (e.g. "llamacpp", "llamacpp-hip")
	RuntimeType string // adapter type (config.RuntimeLlamaCpp | config.RuntimeFake)

	Path         string
	MMProj       string
	Capabilities []string
	Ctx          int
	Parallel     int
	Args         []string
	Aliases      []string

	Pinned  bool
	Preload bool

	// Fake-runtime knobs.
	FakeVRAMMB int
	FakeLoad   time.Duration
}

// FromConfig materialises Specs for every configured model.
func FromConfig(cfg *config.Config) map[string]*Spec {
	out := make(map[string]*Spec, len(cfg.Models))
	for id, m := range cfg.Models {
		out[id] = &Spec{
			ID:           id,
			Runtime:      m.Runtime,
			RuntimeType:  cfg.Runtimes[m.Runtime].Type,
			Path:         m.Path,
			MMProj:       m.MMProj,
			Capabilities: append([]string(nil), m.Capabilities...),
			Ctx:          m.Ctx,
			Parallel:     m.Parallel,
			Args:         append([]string(nil), m.Args...),
			Aliases:      append([]string(nil), m.Aliases...),
			Pinned:       m.Pinned,
			Preload:      m.Preload,
			FakeVRAMMB:   m.FakeVRAMMB,
			FakeLoad:     m.FakeLoad,
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

// ProfileKey identifies the (variant, binary, gpu) tuple a Profile belongs
// to. binaryID should encode path+mtime of the runtime binary so that a
// rebuilt llama.cpp invalidates old measurements; gpuName is the device
// name reported by the monitor.
func (s *Spec) ProfileKey(binaryID, gpuName string) string {
	args := append([]string(nil), s.Args...)
	sort.Strings(args)
	parts := []string{
		s.RuntimeType, s.Path, s.MMProj,
		fmt.Sprint(s.Ctx), fmt.Sprint(s.Parallel),
		strings.Join(args, "\x00"),
		binaryID, gpuName,
	}
	sum := sha256.Sum256([]byte(strings.Join(parts, "\x01")))
	return hex.EncodeToString(sum[:8])
}

// EstimateVRAMMB is the pre-measurement guess used for admission until a
// Profile exists. It is deliberately pessimistic.
//
// TODO(M2): read GGUF metadata (block_count, head_count_kv, head_dim) for a
// real KV-cache estimate instead of the file-size heuristic.
func (s *Spec) EstimateVRAMMB() int {
	if s.RuntimeType == config.RuntimeFake {
		return s.FakeVRAMMB
	}
	var weights int64
	if st, err := os.Stat(s.Path); err == nil {
		weights = st.Size()
	}
	if s.MMProj != "" {
		if st, err := os.Stat(s.MMProj); err == nil {
			weights += st.Size()
		}
	}
	if weights == 0 {
		return 0 // unknown; caller must treat as "cannot admit without measuring"
	}
	// Weights + ~15% for compute buffers and CUDA context.
	est := float64(weights) * 1.15
	// KV cache: ~160 KB/token for a 14B GQA model in f16; scale with size but
	// never assume less than 64 KB/token.
	kvPerTok := max(float64(weights)/60000.0, 64*1024)
	est += kvPerTok * float64(s.Ctx*s.Parallel)
	return int(est / (1024 * 1024))
}
