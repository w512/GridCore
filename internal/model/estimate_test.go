package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/w512/gridcore/internal/gguf/gguftest"
)

// Layouts below mirror the real GGUF headers inspected on the test box; the
// expected totals are the measured per-process VRAM from `gridcore bench`
// (RTX 4060 Ti, llama.cpp b11060, flash-attn on). Tolerance is what the
// scheduler can live with before the measured profile takes over.

func within(t *testing.T, name string, got, want, tolPct int) {
	t.Helper()
	diff := (got - want) * 100 / want
	if diff < -tolPct || diff > tolPct {
		t.Errorf("%s: estimate %d MB vs measured %d MB (%+d%%, tolerance ±%d%%)", name, got, want, diff, tolPct)
	} else {
		t.Logf("%s: estimate %d MB vs measured %d MB (%+d%%)", name, got, want, diff)
	}
}

func qwen3_14b(t *testing.T, dir string) string {
	var w gguftest.Writer
	w.String("general.architecture", "qwen3")
	w.U32("qwen3.block_count", 40)
	w.U32("qwen3.embedding_length", 5120)
	w.U32("qwen3.attention.head_count", 40)
	w.U32("qwen3.attention.head_count_kv", 8)
	w.U32("qwen3.attention.key_length", 128)
	w.U32("qwen3.attention.value_length", 128)
	w.Tensor("token_embd.weight", gguftest.Q4_K, 5120, 151936)
	w.Tensor("output.weight", gguftest.Q6_K, 5120, 151936)
	// 40 blocks; sizes chosen so total non-embedding weight ≈ 8.5 GB like the real file.
	for i := 0; i < 40; i++ {
		w.Tensor(blk(i, "attn_q.weight"), gguftest.Q4_K, 5120, 5120)
		w.Tensor(blk(i, "attn_k.weight"), gguftest.Q4_K, 5120, 1024)
		w.Tensor(blk(i, "attn_v.weight"), gguftest.Q6_K, 5120, 1024)
		w.Tensor(blk(i, "attn_output.weight"), gguftest.Q4_K, 5120, 5120)
		w.Tensor(blk(i, "ffn_gate.weight"), gguftest.Q4_K, 5120, 17408)
		w.Tensor(blk(i, "ffn_up.weight"), gguftest.Q4_K, 5120, 17408)
		w.Tensor(blk(i, "ffn_down.weight"), gguftest.Q6_K, 17408, 5120)
	}
	return w.WriteFile(t, dir, "qwen3-14b.gguf")
}

func blk(i int, name string) string { return "blk." + itoa(i) + "." + name }

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

func TestEstimateQwen3_14B(t *testing.T) {
	dir := t.TempDir()
	sp := &Spec{ID: "qwen3-14b", RuntimeType: "llamacpp", Path: qwen3_14b(t, dir), Ctx: 8192, Parallel: 2}
	e := EstimateVRAM(sp)
	t.Log(e)
	if e.Method != "gguf:qwen3" {
		t.Fatalf("method = %s", e.Method)
	}
	// KV: 40 layers x 8 heads x (128+128) x 2 B x 8192 tokens = 1280 MB
	if e.KVMB < 1270 || e.KVMB > 1290 {
		t.Errorf("kv = %d MB, want ~1280", e.KVMB)
	}
	if e.EmbedMB < 400 {
		t.Errorf("token_embd should be excluded from weights: embed=%d", e.EmbedMB)
	}
	within(t, "qwen3-14b", e.TotalMB, 9732, 12)
}

func TestEstimateGemma4E2B(t *testing.T) {
	// 35 layers; pattern TTTTF (4 sliding, 1 global); last 20 share KV with
	// earlier layers; 1 KV head; k/v 512 global, 256 in windows; window 512.
	var w gguftest.Writer
	w.String("general.architecture", "gemma4")
	w.U32("gemma4.block_count", 35)
	w.U32("gemma4.embedding_length", 1536)
	w.U32("gemma4.attention.head_count", 8)
	w.U32("gemma4.attention.head_count_kv", 1)
	w.U32("gemma4.attention.key_length", 512)
	w.U32("gemma4.attention.value_length", 512)
	w.U32("gemma4.attention.key_length_swa", 256)
	w.U32("gemma4.attention.value_length_swa", 256)
	w.U32("gemma4.attention.sliding_window", 512)
	w.U32("gemma4.attention.shared_kv_layers", 20)
	w.U32("gemma4.embedding_length_per_layer_input", 256)
	pattern := make([]bool, 35)
	for i := range pattern {
		pattern[i] = (i+1)%5 != 0
	}
	w.BoolArray("gemma4.attention.sliding_window_pattern", pattern)
	w.Tensor("token_embd.weight", gguftest.Q6_K, 1536, 262144)
	w.Tensor("per_layer_token_embd.weight", gguftest.Q6_K, 8960, 262144)
	for i := 0; i < 35; i++ {
		ffn := uint64(6144)
		if i >= 15 {
			ffn = 12288
		}
		w.Tensor(blk(i, "attn_q.weight"), gguftest.Q4_0, 1536, 4096)
		w.Tensor(blk(i, "attn_k.weight"), gguftest.Q4_0, 1536, 512)
		w.Tensor(blk(i, "attn_v.weight"), gguftest.Q4_0, 1536, 512)
		w.Tensor(blk(i, "attn_output.weight"), gguftest.Q4_0, 4096, 1536)
		w.Tensor(blk(i, "ffn_gate.weight"), gguftest.Q4_0, 1536, ffn)
		w.Tensor(blk(i, "ffn_up.weight"), gguftest.Q4_0, 1536, ffn)
		w.Tensor(blk(i, "ffn_down.weight"), gguftest.Q4_0, ffn, 1536)
	}
	dir := t.TempDir()
	path := w.WriteFile(t, dir, "gemma4-e2b.gguf")
	mmproj := filepath.Join(dir, "mmproj.gguf")
	if err := os.WriteFile(mmproj, make([]byte, 0), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(mmproj, 990*mb); err != nil {
		t.Fatal(err)
	}

	sp := &Spec{ID: "gemma4-e2b", RuntimeType: "llamacpp", Path: path, MMProj: mmproj, Ctx: 16384, Parallel: 4}
	e := EstimateVRAM(sp)
	t.Log(e)
	// Only 15 layers own KV: 3 global (16384 tok x 2 KB) + 12 windowed
	// (512*4+512 = 2560 tok x 1 KB) ≈ 96 + 30 = 126 MB.
	if e.KVMB < 110 || e.KVMB > 145 {
		t.Errorf("kv = %d MB, want ~126", e.KVMB)
	}
	if e.EmbedMB < 1800 || e.EmbedMB > 2000 {
		t.Errorf("per-layer embeddings must be excluded but token_embd counted (tied): embed=%d", e.EmbedMB)
	}
	if !hasNote(e, "tied") {
		t.Errorf("gemma has no output.weight; expected tied-embeddings note, got %v", e.Notes)
	}
	within(t, "gemma4-e2b", e.TotalMB, 2924, 15)

	// Unified memory: the process footprint measured on an M4 Pro
	// (llama.cpp b11146, --load-mode none) is the device part plus the
	// host side: per-layer embeddings and the input copy of token_embd.
	sp.Unified = true
	u := EstimateVRAM(sp)
	t.Log(u)
	if u.OverheadMB != metalOverheadMB || !hasNote(u, "unified memory") {
		t.Errorf("unified estimate should use the Metal overhead and say so: %s", u)
	}
	within(t, "gemma4-e2b unified", u.TotalMB, 4865, 10)
}

func hasNote(e Estimate, sub string) bool {
	for _, n := range e.Notes {
		if strings.Contains(n, sub) {
			return true
		}
	}
	return false
}

func TestEstimateEncoderModel(t *testing.T) {
	// nomic-bert: no output.weight, causal=false -> no logits, not tied.
	var w gguftest.Writer
	w.String("general.architecture", "nomic-bert")
	w.U32("nomic-bert.block_count", 12)
	w.U32("nomic-bert.embedding_length", 768)
	w.U32("nomic-bert.attention.head_count", 12)
	w.Bool("nomic-bert.attention.causal", false)
	w.Tensor("token_embd.weight", gguftest.F16, 768, 30522)
	for i := 0; i < 12; i++ { // SwiGLU BERT: qkv, out, gate/up/down
		w.Tensor(blk(i, "attn_qkv.weight"), gguftest.F16, 768, 2304)
		w.Tensor(blk(i, "attn_output.weight"), gguftest.F16, 768, 768)
		w.Tensor(blk(i, "ffn_gate.weight"), gguftest.F16, 768, 3072)
		w.Tensor(blk(i, "ffn_up.weight"), gguftest.F16, 768, 3072)
		w.Tensor(blk(i, "ffn_down.weight"), gguftest.F16, 3072, 768)
	}
	path := w.WriteFile(t, t.TempDir(), "nomic.gguf")
	e := EstimateVRAM(&Spec{RuntimeType: "llamacpp", Path: path, Ctx: 2048, Parallel: 4})
	t.Log(e)
	if hasNote(e, "tied") {
		t.Error("encoder must not be treated as tied")
	}
	if e.OverheadMB != cudaContextMB {
		t.Errorf("encoder has no logits buffer: overhead=%d", e.OverheadMB)
	}
	// 12 layers x 12 heads x (64+64) x 2 B x 2048 = 72 MB
	if e.KVMB < 70 || e.KVMB > 76 {
		t.Errorf("kv = %d, want ~72", e.KVMB)
	}
	within(t, "nomic-embed", e.TotalMB, 396, 15)

	u := EstimateVRAM(&Spec{RuntimeType: "llamacpp", Path: path, Ctx: 2048, Parallel: 4, Unified: true})
	if u.OverheadMB != cudaContextMB {
		t.Errorf("encoder overhead must not change on unified memory: %d", u.OverheadMB)
	}
	within(t, "nomic-embed unified", u.TotalMB, 415, 15)
}

func TestEstimateHybridQwen35(t *testing.T) {
	// Ornith-1.5-9B: 33 layers, every 4th is full attention, others SSM.
	var w gguftest.Writer
	w.String("general.architecture", "qwen35")
	w.U32("qwen35.block_count", 33)
	w.U32("qwen35.embedding_length", 4096)
	w.U32("qwen35.attention.head_count", 16)
	w.U32("qwen35.attention.head_count_kv", 4)
	w.U32("qwen35.attention.key_length", 256)
	w.U32("qwen35.attention.value_length", 256)
	w.U32("qwen35.full_attention_interval", 4)
	w.U32("qwen35.ssm.inner_size", 4096)
	w.U32("qwen35.ssm.state_size", 128)
	w.U32("qwen35.ssm.conv_kernel", 4)
	w.Tensor("token_embd.weight", gguftest.Q4_K, 4096, 248320)
	w.Tensor("output.weight", gguftest.Q6_K, 4096, 248320)
	for i := 0; i < 33; i++ {
		w.Tensor(blk(i, "ffn_gate.weight"), gguftest.Q4_K, 4096, 12288)
		w.Tensor(blk(i, "ffn_up.weight"), gguftest.Q4_K, 4096, 12288)
		w.Tensor(blk(i, "ffn_down.weight"), gguftest.Q6_K, 12288, 4096)
		if (i+1)%4 == 0 {
			w.Tensor(blk(i, "attn_q.weight"), gguftest.Q4_K, 4096, 4096)
			w.Tensor(blk(i, "attn_k.weight"), gguftest.Q4_K, 4096, 1024)
			w.Tensor(blk(i, "attn_v.weight"), gguftest.Q4_K, 4096, 1024)
			w.Tensor(blk(i, "attn_output.weight"), gguftest.Q4_K, 4096, 4096)
		} else {
			w.Tensor(blk(i, "attn_qkv.weight"), gguftest.Q4_K, 4096, 12288)
			w.Tensor(blk(i, "ssm_out.weight"), gguftest.Q4_K, 4096, 4096)
			w.Tensor(blk(i, "attn_gate.weight"), gguftest.Q4_K, 4096, 4096)
		}
	}
	dir := t.TempDir()
	path := w.WriteFile(t, dir, "ornith.gguf")
	mmproj := filepath.Join(dir, "mmproj.gguf")
	_ = os.WriteFile(mmproj, nil, 0o644)
	_ = os.Truncate(mmproj, 920*mb)

	sp := &Spec{ID: "ornith-9b", RuntimeType: "llamacpp", Path: path, MMProj: mmproj, Ctx: 16384, Parallel: 2}
	e := EstimateVRAM(sp)
	t.Log(e)
	// 8 attention layers x 4 heads x 512 x 2 B x 16384 = 512 MB, plus small SSM state.
	if e.KVMB < 500 || e.KVMB > 620 {
		t.Errorf("kv = %d MB, want ~512 + ssm", e.KVMB)
	}
	within(t, "ornith-9b", e.TotalMB, 6814, 15)
}

func TestEstimateCacheTypes(t *testing.T) {
	dir := t.TempDir()
	path := qwen3_14b(t, dir)
	f16 := EstimateVRAM(&Spec{RuntimeType: "llamacpp", Path: path, Ctx: 8192, Parallel: 1})
	q8 := EstimateVRAM(&Spec{RuntimeType: "llamacpp", Path: path, Ctx: 8192, Parallel: 1, Args: []string{"-ctk", "q8_0", "--cache-type-v=q8_0"}})
	if q8.KVMB >= f16.KVMB || q8.KVMB < f16.KVMB/2 {
		t.Errorf("q8_0 cache should be ~half of f16: %d vs %d", q8.KVMB, f16.KVMB)
	}
	if f16.KVMB != EstimateVRAM(&Spec{RuntimeType: "llamacpp", Path: path, Ctx: 8192, Parallel: 4}).KVMB {
		t.Error("ctx is the shared pool; parallel must not change full-attention KV size")
	}
}

func TestEstimateFallbacks(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.gguf")
	_ = os.WriteFile(bad, []byte("not a gguf file at all, but 9 GB in spirit"), 0o644)
	e := EstimateVRAM(&Spec{RuntimeType: "llamacpp", Path: bad, Ctx: 4096, Parallel: 1})
	if e.Method != "filesize" || len(e.Notes) == 0 {
		t.Errorf("unreadable header should fall back: %+v", e)
	}
	e = EstimateVRAM(&Spec{RuntimeType: "llamacpp", Path: filepath.Join(dir, "missing.gguf"), Ctx: 4096, Parallel: 1})
	if e.TotalMB != 0 {
		t.Errorf("missing file should estimate 0, got %d", e.TotalMB)
	}
	e = EstimateVRAM(&Spec{RuntimeType: "simulation", SimulatedVRAMMB: 777})
	if e.TotalMB != 777 || e.Method != "simulation" {
		t.Errorf("simulation = %+v", e)
	}
}
