package model

import (
	"fmt"
	"os"
	"strings"

	"github.com/gridcore/gridcore/internal/gguf"
)

// Estimate is a pre-measurement VRAM prediction with its breakdown, so
// `gridcore models --explain` and error messages can show where the number
// comes from.
type Estimate struct {
	TotalMB    int
	WeightsMB  int // tensors that live in VRAM (embedding tables excluded)
	EmbedMB    int // token / per-layer embeddings kept in host memory
	KVMB       int
	MMProjMB   int
	OverheadMB int // CUDA context + compute/logit buffers
	Method     string
	Notes      []string
}

// EstimateVRAM predicts sp's footprint. It reads the GGUF header when
// possible and falls back to the file-size heuristic otherwise.
func EstimateVRAM(sp *Spec) Estimate {
	if sp.RuntimeType == "fake" {
		return Estimate{TotalMB: sp.FakeVRAMMB, Method: "fake"}
	}
	est, err := estimateFromGGUF(sp)
	if err == nil {
		return est
	}
	fb := estimateFromFileSize(sp)
	fb.Notes = append(fb.Notes, "gguf header unreadable: "+err.Error())
	return fb
}

const (
	mb            = 1024 * 1024
	cudaContextMB = 128 // driver context per process
	nUbatch       = 512 // llama-server default; logits buffer is n_ubatch x vocab x f32
	swaPad        = 512 // llama.cpp pads the SWA cache by about n_ubatch tokens
	mmprojBufMB   = 256 // vision/audio encoder compute buffers
)

func estimateFromGGUF(sp *Spec) (Estimate, error) {
	f, err := gguf.Read(sp.Path)
	if err != nil {
		return Estimate{}, err
	}
	var e Estimate
	e.Method = "gguf:" + f.Arch()

	// Weights: everything except embedding tables, which llama.cpp keeps in
	// host memory (token_embd is the CPU-side input layer; Gemma's
	// per_layer_token_embd is read lazily). Exception: models with tied
	// embeddings (no output.weight) duplicate token_embd onto the GPU as the
	// output projection, so it counts.
	_, hasOutput := f.Tensor("output.weight")
	encoder := isEncoder(f)
	tied := !hasOutput && !encoder
	if tied {
		e.Notes = append(e.Notes, "tied embeddings: token_embd counted in VRAM")
	}
	var vocab uint64
	for _, t := range f.Tensors {
		b := t.Bytes()
		if t.Name == "token_embd.weight" && len(t.Dims) == 2 {
			vocab = t.Dims[1]
		}
		if isHostEmbedding(t.Name) && !(tied && t.Name == "token_embd.weight") {
			e.EmbedMB += int(b / mb)
			continue
		}
		e.WeightsMB += int(b / mb)
	}
	if vocab == 0 {
		if v, ok := f.Uint("vocab_size"); ok {
			vocab = v
		} else {
			vocab = 32768
			e.Notes = append(e.Notes, "vocab size unknown; assuming 32K for logits buffer")
		}
	}

	// KV cache.
	kvBytes, notes := kvCacheBytes(f, sp)
	e.KVMB = int(kvBytes / mb)
	e.Notes = append(e.Notes, notes...)

	// Multimodal projector: fully resident plus encoder buffers.
	if sp.MMProj != "" {
		if st, err := os.Stat(sp.MMProj); err == nil {
			e.MMProjMB = int(st.Size()/mb) + mmprojBufMB
		} else {
			e.Notes = append(e.Notes, "mmproj not found: "+err.Error())
		}
	}

	e.OverheadMB = cudaContextMB
	if !encoder {
		e.OverheadMB += int(uint64(nUbatch) * vocab * 4 / mb) // logits
	}
	e.TotalMB = e.WeightsMB + e.KVMB + e.MMProjMB + e.OverheadMB
	return e, nil
}

// isEncoder reports embedding-only models (BERT family): no logits, no
// output projection.
func isEncoder(f *gguf.File) bool {
	if v, ok := f.KV[f.Arch()+".attention.causal"].(bool); ok && !v {
		return true
	}
	return strings.Contains(f.Arch(), "bert")
}

func isHostEmbedding(name string) bool {
	return name == "token_embd.weight" || name == "per_layer_token_embd.weight" ||
		strings.HasPrefix(name, "token_types") || name == "position_embd.weight"
}

// kvCacheBytes sizes the KV cache for ctx tokens, walking the layer layout:
// sliding-window layers hold only the window, shared-KV layers hold nothing,
// hybrid (SSM / linear attention) layers hold a small recurrent state.
func kvCacheBytes(f *gguf.File, sp *Spec) (uint64, []string) {
	var notes []string
	nLayer, ok := f.Uint("block_count")
	if !ok || nLayer == 0 {
		return 0, []string{"block_count missing; KV cache not estimated"}
	}
	n := int(nLayer)
	ctx := uint64(sp.Ctx)

	headCount, _ := f.Uint("attention.head_count")
	embd, _ := f.Uint("embedding_length")
	kvHeads, ok := f.Uints("attention.head_count_kv", n)
	if !ok {
		kvHeads = make([]uint64, n)
		for i := range kvHeads {
			kvHeads[i] = headCount
		}
	}
	if len(kvHeads) < n {
		last := kvHeads[len(kvHeads)-1]
		for len(kvHeads) < n {
			kvHeads = append(kvHeads, last)
		}
	}
	keyLen, ok := f.Uint("attention.key_length")
	if !ok && headCount > 0 {
		keyLen = embd / headCount
	}
	valLen, ok := f.Uint("attention.value_length")
	if !ok {
		valLen = keyLen
	}
	keyLenSWA, ok := f.Uint("attention.key_length_swa")
	if !ok {
		keyLenSWA = keyLen
	}
	valLenSWA, ok := f.Uint("attention.value_length_swa")
	if !ok {
		valLenSWA = valLen
	}

	// Which layers are sliding-window.
	swa := make([]bool, n)
	if nSWA, ok := f.Uint("attention.sliding_window"); ok && nSWA > 0 && nSWA < ctx {
		if pattern, ok := f.Bools("attention.sliding_window_pattern"); ok && len(pattern) > 0 {
			for i := range swa {
				swa[i] = pattern[i%len(pattern)]
			}
		} else if period, ok := f.Uint("attention.sliding_window_pattern"); ok && period > 0 {
			// Gemma 3 style: every period-th layer is global.
			for i := range swa {
				swa[i] = uint64(i+1)%period != 0
			}
		} else {
			// Window declared but no pattern: assume all layers use it
			// (Mistral-style). Pessimistic if wrong? No — optimistic, so
			// note it.
			for i := range swa {
				swa[i] = true
			}
			notes = append(notes, "sliding window without layer pattern; assuming all layers windowed")
		}
	}
	nSWA, _ := f.Uint("attention.sliding_window")

	// Layers without their own cache.
	shared, _ := f.Uint("attention.shared_kv_layers")    // Gemma 4: last N layers reuse KV
	fullInterval, _ := f.Uint("full_attention_interval") // Qwen3.5 / Qwen3-Next hybrid

	bytesK, bytesV := cacheTypeBytes(sp.Args)
	var total float64
	var recurrent int
	for i := 0; i < n; i++ {
		if shared > 0 && uint64(i) >= nLayer-shared {
			continue
		}
		if fullInterval > 0 && uint64(i+1)%fullInterval != 0 {
			recurrent++
			continue
		}
		if kvHeads[i] == 0 {
			continue
		}
		tokens := ctx
		kl, vl := keyLen, valLen
		if swa[i] {
			kl, vl = keyLenSWA, valLenSWA
			if w := nSWA*uint64(sp.Parallel) + swaPad; w < tokens {
				tokens = w
			}
		}
		perTok := float64(kvHeads[i]) * (float64(kl)*bytesK + float64(vl)*bytesV)
		total += perTok * float64(tokens)
	}

	// Recurrent state for hybrid layers: per slot, inner_size x state_size f32
	// plus the conv window.
	if recurrent > 0 {
		inner, _ := f.Uint("ssm.inner_size")
		state, _ := f.Uint("ssm.state_size")
		conv, _ := f.Uint("ssm.conv_kernel")
		perLayerSlot := float64(inner*state*4) + float64(inner*conv*4)
		total += perLayerSlot * float64(recurrent) * float64(sp.Parallel)
	}
	return uint64(total), notes
}

// cacheTypeBytes reads -ctk/-ctv from the model args; default f16.
func cacheTypeBytes(args []string) (k, v float64) {
	k, v = 2, 2
	get := func(i int, flag string) (string, bool) {
		a := args[i]
		if val, ok := strings.CutPrefix(a, flag+"="); ok {
			return val, true
		}
		if a == flag && i+1 < len(args) {
			return args[i+1], true
		}
		return "", false
	}
	for i := range args {
		for _, flag := range []string{"-ctk", "--cache-type-k"} {
			if val, ok := get(i, flag); ok {
				if b := gguf.CacheTypeBytes(val); b > 0 {
					k = b
				}
			}
		}
		for _, flag := range []string{"-ctv", "--cache-type-v"} {
			if val, ok := get(i, flag); ok {
				if b := gguf.CacheTypeBytes(val); b > 0 {
					v = b
				}
			}
		}
	}
	return k, v
}

// estimateFromFileSize is the pre-GGUF heuristic, kept as a fallback.
func estimateFromFileSize(sp *Spec) Estimate {
	e := Estimate{Method: "filesize"}
	var weights int64
	if st, err := os.Stat(sp.Path); err == nil {
		weights = st.Size()
	}
	if sp.MMProj != "" {
		if st, err := os.Stat(sp.MMProj); err == nil {
			weights += st.Size()
		}
	}
	if weights == 0 {
		e.Notes = append(e.Notes, "model file not found; no estimate")
		return e
	}
	e.WeightsMB = int(float64(weights) * 1.10 / mb)
	kvPerTok := max(float64(weights)/60000.0, 64*1024)
	e.KVMB = int(kvPerTok * float64(sp.Ctx) / mb)
	e.TotalMB = e.WeightsMB + e.KVMB
	return e
}

// String renders the breakdown on one line.
func (e Estimate) String() string {
	s := fmt.Sprintf("%d MB (%s: weights %d + kv %d + mmproj %d + overhead %d; host embeddings %d)",
		e.TotalMB, e.Method, e.WeightsMB, e.KVMB, e.MMProjMB, e.OverheadMB, e.EmbedMB)
	if len(e.Notes) > 0 {
		s += " [" + strings.Join(e.Notes, "; ") + "]"
	}
	return s
}
