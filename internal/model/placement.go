package model

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// placement is what the llama-server flags in a model's args say about
// which tensors stay in host memory. The zero value means everything goes
// to the GPU, which is what the runtime's default --n-gpu-layers 999 does.
//
// llama.cpp orders the offloadable "layers" as blk.0 .. blk.(n_layer_all-1)
// followed by the output layer and gives the GPU the *last* N of them, so
// the output projection is offloaded first and the low blocks are the ones
// left on the CPU. n_layer_all includes MTP (nextn) blocks, which are never
// loaded at all. Verified against `load_tensors:` output of b11060.
type placement struct {
	gpuLayers   int              // -ngl N; -1 = not set (all)
	cpuMoE      int              // --n-cpu-moe N; -1 = --cpu-moe (all); 0 = none
	cpuPatterns []*regexp.Regexp // -ot <regex>=CPU
	notes       []string
}

// parsePlacement reads the offload flags out of args. Unknown values are
// ignored with a note rather than failing: an estimate is advisory.
func parsePlacement(args []string) placement {
	pl := placement{gpuLayers: -1}
	get := func(i int, flags ...string) (string, bool) {
		for _, flag := range flags {
			if val, ok := strings.CutPrefix(args[i], flag+"="); ok {
				return val, true
			}
			if args[i] == flag && i+1 < len(args) {
				return args[i+1], true
			}
		}
		return "", false
	}
	for i := range args {
		if val, ok := get(i, "-ngl", "--gpu-layers", "--n-gpu-layers"); ok {
			if n, err := strconv.Atoi(val); err == nil && n >= 0 {
				pl.gpuLayers = n
			} else {
				pl.notes = append(pl.notes, fmt.Sprintf("-ngl %q not modelled; assuming all layers on GPU", val))
			}
		}
		if args[i] == "--cpu-moe" || args[i] == "-cmoe" {
			pl.cpuMoE = -1
		}
		if val, ok := get(i, "-ncmoe", "--n-cpu-moe"); ok {
			if n, err := strconv.Atoi(val); err == nil && n >= 0 && pl.cpuMoE != -1 {
				pl.cpuMoE = n
			}
		}
		if val, ok := get(i, "-ot", "--override-tensor"); ok {
			for _, rule := range strings.Split(val, ",") {
				pat, buf, found := strings.Cut(rule, "=")
				if !found {
					continue
				}
				if !strings.EqualFold(strings.TrimSpace(buf), "CPU") {
					pl.notes = append(pl.notes, fmt.Sprintf("-ot %s: only =CPU is modelled", rule))
					continue
				}
				re, err := regexp.Compile(strings.TrimSpace(pat))
				if err != nil {
					pl.notes = append(pl.notes, fmt.Sprintf("-ot %s: %v", rule, err))
					continue
				}
				pl.cpuPatterns = append(pl.cpuPatterns, re)
			}
		}
	}
	return pl
}

// firstGPUBlock returns the lowest block index that lives on the GPU and
// whether the output layer does, given the block count including MTP blocks.
func (pl placement) firstGPUBlock(nLayerAll int) (first int, outputOnGPU bool) {
	if pl.gpuLayers < 0 || pl.gpuLayers > nLayerAll {
		return 0, true
	}
	if pl.gpuLayers == 0 {
		return nLayerAll, false
	}
	// N layers on the GPU: the output plus the top N-1 blocks.
	return nLayerAll + 1 - pl.gpuLayers, true
}

// expertsOnCPU reports whether the expert tensors of block i stay on the host.
func (pl placement) expertsOnCPU(i int) bool {
	return pl.cpuMoE == -1 || i < pl.cpuMoE
}

// overriddenToCPU reports whether an explicit -ot rule pins name to the host.
func (pl placement) overriddenToCPU(name string) bool {
	for _, re := range pl.cpuPatterns {
		if re.MatchString(name) {
			return true
		}
	}
	return false
}

// blockIndex parses "blk.<i>." prefixes; ok is false for non-block tensors.
func blockIndex(name string) (int, bool) {
	rest, ok := strings.CutPrefix(name, "blk.")
	if !ok {
		return 0, false
	}
	idx, _, ok := strings.Cut(rest, ".")
	if !ok {
		return 0, false
	}
	i, err := strconv.Atoi(idx)
	if err != nil || i < 0 {
		return 0, false
	}
	return i, true
}

// isExpertTensor matches the routed-expert weights of MoE blocks
// (ffn_gate_exps / ffn_up_exps / ffn_down_exps). Shared experts (_shexp)
// are dense and stay with the block.
func isExpertTensor(name string) bool {
	return strings.Contains(name, "_exps.")
}
