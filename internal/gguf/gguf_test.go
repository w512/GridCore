package gguf

import (
	"bytes"
	"testing"

	"github.com/w512/gridcore/internal/gguf/gguftest"
)

func TestParseRoundTrip(t *testing.T) {
	var w gguftest.Writer
	w.String("general.architecture", "qwen3")
	w.String("general.name", "Test 14B")
	w.U32("qwen3.block_count", 40)
	w.U32("qwen3.attention.head_count_kv", 8)
	w.F32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
	w.Bool("tokenizer.ggml.add_bos_token", false)
	w.I32Array("gemma4.attention.head_count_kv", []int32{4, 4, 0, 4})
	w.BoolArray("gemma4.attention.sliding_window_pattern", []bool{true, true, false})
	big := make([]string, maxArrayKeep+10)
	for i := range big {
		big[i] = "tok"
	}
	w.StringArray("tokenizer.ggml.tokens", big)
	bigF := make([]float32, maxArrayKeep+10)
	w.F32Array("tokenizer.ggml.scores", bigF)
	w.U32("after.skip", 7) // must still be readable after the skipped arrays
	w.Tensor("token_embd.weight", gguftest.Q4_K, 5120, 151936)
	w.Tensor("blk.0.attn_q.weight", gguftest.Q4_K, 5120, 5120)
	w.Tensor("output_norm.weight", gguftest.F32, 5120)

	f, err := Parse(bytes.NewReader(w.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if f.Arch() != "qwen3" || f.Name() != "Test 14B" || f.Version != 3 {
		t.Errorf("header = %s %s v%d", f.Arch(), f.Name(), f.Version)
	}
	if v, ok := f.Uint("block_count"); !ok || v != 40 {
		t.Errorf("block_count = %d %v", v, ok)
	}
	if v, ok := f.Uint("attention.head_count_kv"); !ok || v != 8 {
		t.Errorf("head_count_kv = %d %v", v, ok)
	}
	if v, ok := f.KV["tokenizer.ggml.add_bos_token"].(bool); !ok || v {
		t.Errorf("bool = %v", f.KV["tokenizer.ggml.add_bos_token"])
	}
	if sk, ok := f.KV["tokenizer.ggml.tokens"].(Skipped); !ok || sk.Len != maxArrayKeep+10 {
		t.Errorf("large string array should be Skipped, got %v", f.KV["tokenizer.ggml.tokens"])
	}
	if sk, ok := f.KV["tokenizer.ggml.scores"].(Skipped); !ok || sk.Len != maxArrayKeep+10 {
		t.Errorf("large float array should be Skipped, got %v", f.KV["tokenizer.ggml.scores"])
	}
	if v, ok := f.KV["after.skip"]; !ok || v.(uint64) != 7 {
		t.Errorf("value after skipped arrays = %v", v)
	}
	if len(f.Tensors) != 3 {
		t.Fatalf("tensors = %d", len(f.Tensors))
	}
	te, _ := f.Tensor("token_embd.weight")
	if te.Bytes() != 5120*151936*144/256 {
		t.Errorf("Q4_K bytes = %d", te.Bytes())
	}
	on, _ := f.Tensor("output_norm.weight")
	if on.Bytes() != 5120*4 {
		t.Errorf("F32 bytes = %d", on.Bytes())
	}

	// Arrays via a different arch prefix.
	f.KV["general.architecture"] = "gemma4"
	heads, ok := f.Uints("attention.head_count_kv", 4)
	if !ok || len(heads) != 4 || heads[2] != 0 {
		t.Errorf("per-layer heads = %v %v", heads, ok)
	}
	pat, ok := f.Bools("attention.sliding_window_pattern")
	if !ok || len(pat) != 3 || pat[2] {
		t.Errorf("pattern = %v %v", pat, ok)
	}
	// Scalar broadcast.
	f.KV["general.architecture"] = "qwen3"
	heads, _ = f.Uints("attention.head_count_kv", 3)
	if len(heads) != 3 || heads[2] != 8 {
		t.Errorf("broadcast = %v", heads)
	}
}

func TestParseRejectsGarbage(t *testing.T) {
	if _, err := Parse(bytes.NewReader([]byte("GGML....."))); err == nil {
		t.Error("bad magic accepted")
	}
	var w gguftest.Writer
	b := w.Bytes()
	b[4] = 9 // version 9
	if _, err := Parse(bytes.NewReader(b)); err == nil {
		t.Error("bad version accepted")
	}
	if _, err := Parse(bytes.NewReader([]byte("GGUF"))); err == nil {
		t.Error("truncated accepted")
	}
}

func TestReadFromFileSkipsWithSeek(t *testing.T) {
	var w gguftest.Writer
	w.String("general.architecture", "x")
	big := make([]float32, 300000) // 1.2 MB, larger than the bufio buffer
	w.F32Array("tokenizer.ggml.scores", big)
	w.U32("x.block_count", 3)
	p := w.WriteFile(t, t.TempDir(), "m.gguf")
	f, err := Read(p)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := f.Uint("block_count"); v != 3 {
		t.Errorf("block_count after 1.2 MB skip = %d", v)
	}
}

func TestCacheTypeBytes(t *testing.T) {
	if CacheTypeBytes("f16") != 2 || CacheTypeBytes("q8_0") <= 1 || CacheTypeBytes("q8_0") >= 1.1 || CacheTypeBytes("nope") != 0 {
		t.Error("cache type sizes")
	}
}
