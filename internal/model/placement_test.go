package model

import (
	"testing"

	"github.com/w512/gridcore/internal/gguf/gguftest"
)

// Offload flags. The layouts and expected numbers follow what llama.cpp
// b11060 printed under `load_tensors:` for the real files on the test box.

func TestParsePlacement(t *testing.T) {
	cases := []struct {
		args     []string
		gpu, moe int
		patterns int
	}{
		{nil, -1, 0, 0},
		{[]string{"--temp", "1.0"}, -1, 0, 0},
		{[]string{"-ngl", "64"}, 64, 0, 0},
		{[]string{"--n-gpu-layers=64"}, 64, 0, 0},
		{[]string{"--gpu-layers", "0"}, 0, 0, 0},
		{[]string{"-ngl", "auto"}, -1, 0, 0}, // not modelled, noted
		{[]string{"--cpu-moe"}, -1, -1, 0},
		{[]string{"-cmoe"}, -1, -1, 0},
		{[]string{"--n-cpu-moe", "12"}, -1, 12, 0},
		{[]string{"-ncmoe=12"}, -1, 12, 0},
		{[]string{"--cpu-moe", "--n-cpu-moe", "3"}, -1, -1, 0}, // all wins
		{[]string{"-ot", `\.ffn_.*_exps\.=CPU`}, -1, 0, 1},
		{[]string{"--override-tensor", "exps=CPU,attn=CUDA1"}, -1, 0, 1},
		{[]string{"-ot", "exps=cpu"}, -1, 0, 1},
	}
	for _, c := range cases {
		pl := parsePlacement(c.args)
		if pl.gpuLayers != c.gpu || pl.cpuMoE != c.moe || len(pl.cpuPatterns) != c.patterns {
			t.Errorf("%v: gpu=%d moe=%d patterns=%d, want %d/%d/%d", c.args, pl.gpuLayers, pl.cpuMoE, len(pl.cpuPatterns), c.gpu, c.moe, c.patterns)
		}
	}
	if pl := parsePlacement([]string{"-ngl", "auto"}); len(pl.notes) == 0 {
		t.Error("unparseable -ngl should leave a note")
	}
}

func TestFirstGPUBlock(t *testing.T) {
	// 65 blocks in the header (64 + 1 MTP), output makes 66 offloadable layers.
	cases := []struct {
		ngl      int
		first    int
		output   bool
		nLayrAll int
	}{
		{-1, 0, true, 65},  // default: everything
		{999, 0, true, 65}, // runtime default
		{66, 0, true, 65},  // exactly all
		{65, 1, true, 65},  // output + 64 blocks -> blk.0 on host
		{64, 2, true, 65},  // what the box ran: blk.0, blk.1 on host
		{1, 65, true, 65},  // output only
		{0, 65, false, 65}, // nothing
		{38, 3, true, 40},  // dense 40-layer model
	}
	for _, c := range cases {
		pl := placement{gpuLayers: c.ngl}
		first, out := pl.firstGPUBlock(c.nLayrAll)
		if first != c.first || out != c.output {
			t.Errorf("ngl %d of %d: first=%d output=%v, want %d/%v", c.ngl, c.nLayrAll, first, out, c.first, c.output)
		}
	}
}

func TestEstimateNGLLeavesLowBlocksOnHost(t *testing.T) {
	dir := t.TempDir()
	path := qwen3_14b(t, dir) // 40 dense blocks, no MTP
	full := EstimateVRAM(&Spec{RuntimeType: "llamacpp", Path: path, Ctx: 8192, Parallel: 2})
	part := EstimateVRAM(&Spec{RuntimeType: "llamacpp", Path: path, Ctx: 8192, Parallel: 2, Args: []string{"-ngl", "38"}})
	t.Log(full)
	t.Log(part)

	// 41 offloadable layers, 38 on the GPU: output + blk.3..39 -> 3 blocks on host.
	perBlock := full.WeightsMB / 41 // rough: output.weight is about one block here
	hostLo, hostHi := 3*perBlock*8/10, 3*perBlock*12/10
	if part.HostWeightsMB < hostLo || part.HostWeightsMB > hostHi {
		t.Errorf("host weights = %d MB, want ~3 blocks (%d..%d)", part.HostWeightsMB, hostLo, hostHi)
	}
	if part.WeightsMB+part.HostWeightsMB != full.WeightsMB {
		t.Errorf("weights must be conserved: %d + %d != %d", part.WeightsMB, part.HostWeightsMB, full.WeightsMB)
	}
	// KV shrinks with the blocks: 37/40 of 1280 MB.
	if want := full.KVMB * 37 / 40; part.KVMB < want-8 || part.KVMB > want+8 {
		t.Errorf("kv = %d MB, want ~%d", part.KVMB, want)
	}
	if !hasNote(part, "3 block(s)") {
		t.Errorf("expected a note about 3 host blocks: %v", part.Notes)
	}
	if part.OverheadMB != full.OverheadMB {
		t.Error("output stays on the GPU for ngl >= 1; logits buffer must remain")
	}

	none := EstimateVRAM(&Spec{RuntimeType: "llamacpp", Path: path, Ctx: 8192, Parallel: 2, Args: []string{"-ngl", "0"}})
	if none.WeightsMB != 0 || none.KVMB != 0 || none.OverheadMB != cudaContextMB {
		t.Errorf("-ngl 0 should leave only the CUDA context: %s", none)
	}
}

// qwen38_27b mirrors the header of Qwen3.8-27B-UD-Q4_K_M: 65 blocks of which
// one is an MTP head (nextn_predict_layers = 1), every 4th real block is full
// attention with 4 KV heads x 256, the rest Gated DeltaNet.
func qwen38_27b(t *testing.T, dir string) string {
	var w gguftest.Writer
	w.String("general.architecture", "qwen35")
	w.U32("qwen35.block_count", 65)
	w.U32("qwen35.nextn_predict_layers", 1)
	w.U32("qwen35.embedding_length", 5120)
	w.U32("qwen35.attention.head_count", 24)
	w.U32("qwen35.attention.head_count_kv", 4)
	w.U32("qwen35.attention.key_length", 256)
	w.U32("qwen35.attention.value_length", 256)
	w.U32("qwen35.full_attention_interval", 4)
	w.U32("qwen35.ssm.inner_size", 6144)
	w.U32("qwen35.ssm.state_size", 128)
	w.U32("qwen35.ssm.conv_kernel", 4)
	w.Tensor("token_embd.weight", gguftest.Q4_K, 5120, 248320)
	w.Tensor("output.weight", gguftest.Q6_K, 5120, 248320)
	for i := 0; i < 65; i++ {
		// UD-Q4_K_M keeps ffn_down at Q6_K in most blocks; the real CUDA0
		// model buffer is 14262 MiB for 62 blocks + output (~213 MB/block).
		down := uint32(gguftest.Q6_K)
		if i%3 == 2 {
			down = gguftest.Q4_K
		}
		w.Tensor(blk(i, "ffn_gate.weight"), gguftest.Q4_K, 5120, 17408)
		w.Tensor(blk(i, "ffn_up.weight"), gguftest.Q4_K, 5120, 17408)
		w.Tensor(blk(i, "ffn_down.weight"), down, 17408, 5120)
		if (i+1)%4 == 0 {
			w.Tensor(blk(i, "attn_q.weight"), gguftest.Q4_K, 5120, 12288)
			w.Tensor(blk(i, "attn_k.weight"), gguftest.Q4_K, 5120, 1024)
			w.Tensor(blk(i, "attn_v.weight"), gguftest.Q4_K, 5120, 1024)
			w.Tensor(blk(i, "attn_output.weight"), gguftest.Q4_K, 6144, 5120)
		} else {
			w.Tensor(blk(i, "attn_qkv.weight"), gguftest.Q4_K, 5120, 8192)
			w.Tensor(blk(i, "ssm_out.weight"), gguftest.Q4_K, 4096, 5120)
			w.Tensor(blk(i, "attn_gate.weight"), gguftest.Q4_K, 5120, 4096)
		}
	}
	w.Tensor(blk(64, "nextn.eh_proj.weight"), gguftest.Q4_K, 10240, 5120)
	return w.WriteFile(t, dir, "qwen38-27b.gguf")
}

func TestEstimateSkipsMTPBlock(t *testing.T) {
	dir := t.TempDir()
	path := qwen38_27b(t, dir)
	full := EstimateVRAM(&Spec{ID: "qwen38-27b", RuntimeType: "llamacpp", Path: path, Ctx: 8192, Parallel: 1})
	t.Log(full)
	if !hasNote(full, "1 MTP block") {
		t.Errorf("expected MTP note: %v", full.Notes)
	}
	// llama.cpp: CUDA0 KV buffer 512 MiB (16 full-attention layers) + RS 143 MiB.
	if full.KVMB < 630 || full.KVMB > 680 {
		t.Errorf("kv = %d MB, want ~655", full.KVMB)
	}
	// Measured all-GPU footprint on the box: 15632 MiB.
	within(t, "qwen38-27b all on GPU", full.TotalMB, 15632, 4)

	// -ngl 64 -> blk.0 and blk.1 on the host; measured 15224-15250 MiB.
	part := EstimateVRAM(&Spec{ID: "qwen38-27b", RuntimeType: "llamacpp", Path: path, Ctx: 8192, Parallel: 1, Args: []string{"--n-gpu-layers", "64"}})
	t.Log(part)
	if !hasNote(part, "2 block(s)") {
		t.Errorf("expected 2 host blocks: %v", part.Notes)
	}
	if part.WeightsMB+part.HostWeightsMB != full.WeightsMB {
		t.Errorf("weights must be conserved: %d + %d != %d", part.WeightsMB, part.HostWeightsMB, full.WeightsMB)
	}
	within(t, "qwen38-27b -ngl 64", part.TotalMB, 15250, 4)
}

// ornith_35b_a3b mirrors Ornith-1.5-35B-A3B (qwen35moe): 40 blocks, 256
// routed experts of intermediate 512 plus a shared expert, hybrid attention
// every 4th block with 2 KV heads x 256.
func ornith_35b_a3b(t *testing.T, dir string) string {
	var w gguftest.Writer
	w.String("general.architecture", "qwen35moe")
	w.U32("qwen35moe.block_count", 40)
	w.U32("qwen35moe.embedding_length", 2048)
	w.U32("qwen35moe.attention.head_count", 16)
	w.U32("qwen35moe.attention.head_count_kv", 2)
	w.U32("qwen35moe.attention.key_length", 256)
	w.U32("qwen35moe.attention.value_length", 256)
	w.U32("qwen35moe.full_attention_interval", 4)
	w.U32("qwen35moe.expert_count", 256)
	w.U32("qwen35moe.expert_used_count", 8)
	w.U32("qwen35moe.ssm.inner_size", 4096)
	w.U32("qwen35moe.ssm.state_size", 128)
	w.U32("qwen35moe.ssm.conv_kernel", 4)
	w.Tensor("token_embd.weight", gguftest.Q4_K, 2048, 248320)
	w.Tensor("output.weight", gguftest.Q6_K, 2048, 248320)
	for i := 0; i < 40; i++ {
		// routed experts: 3D tensors [in, out, n_expert]
		w.Tensor(blk(i, "ffn_gate_exps.weight"), gguftest.Q4_K, 2048, 512, 256)
		w.Tensor(blk(i, "ffn_up_exps.weight"), gguftest.Q4_K, 2048, 512, 256)
		w.Tensor(blk(i, "ffn_down_exps.weight"), gguftest.Q6_K, 512, 2048, 256)
		w.Tensor(blk(i, "ffn_gate_inp.weight"), gguftest.F32, 2048, 256)
		// shared expert stays dense
		w.Tensor(blk(i, "ffn_gate_shexp.weight"), gguftest.Q4_K, 2048, 512)
		w.Tensor(blk(i, "ffn_up_shexp.weight"), gguftest.Q4_K, 2048, 512)
		w.Tensor(blk(i, "ffn_down_shexp.weight"), gguftest.Q6_K, 512, 2048)
		if (i+1)%4 == 0 {
			w.Tensor(blk(i, "attn_q.weight"), gguftest.Q4_K, 2048, 8192)
			w.Tensor(blk(i, "attn_k.weight"), gguftest.Q4_K, 2048, 512)
			w.Tensor(blk(i, "attn_v.weight"), gguftest.Q4_K, 2048, 512)
			w.Tensor(blk(i, "attn_output.weight"), gguftest.Q4_K, 4096, 2048)
		} else {
			w.Tensor(blk(i, "attn_qkv.weight"), gguftest.Q4_K, 2048, 8192)
			w.Tensor(blk(i, "ssm_out.weight"), gguftest.Q4_K, 4096, 2048)
			w.Tensor(blk(i, "attn_gate.weight"), gguftest.Q4_K, 2048, 4096)
		}
	}
	return w.WriteFile(t, dir, "ornith-35b.gguf")
}

func TestEstimateMoEExpertsOnHost(t *testing.T) {
	dir := t.TempDir()
	path := ornith_35b_a3b(t, dir)
	spec := func(args ...string) *Spec {
		return &Spec{ID: "ornith-35b", RuntimeType: "llamacpp", Path: path, Ctx: 8192, Parallel: 1, Args: args}
	}
	full := EstimateVRAM(spec())
	all := EstimateVRAM(spec("--cpu-moe"))
	half := EstimateVRAM(spec("--n-cpu-moe", "20"))
	ot := EstimateVRAM(spec("-ot", `\.ffn_(up|down|gate)_exps\.=CPU`))
	t.Log(full)
	t.Log(all)
	t.Log(half)

	if full.WeightsMB < 18000 {
		t.Errorf("full MoE weights on GPU = %d MB, expected ~19-20 GB", full.WeightsMB)
	}
	if all.WeightsMB+all.HostWeightsMB != full.WeightsMB {
		t.Errorf("weights must be conserved: %d + %d != %d", all.WeightsMB, all.HostWeightsMB, full.WeightsMB)
	}
	// Experts are ~90% of the model; what stays is attention, shared
	// experts, routers and the output projection: well under 3 GB.
	if all.WeightsMB > 3000 || all.WeightsMB < 1000 {
		t.Errorf("--cpu-moe leaves %d MB on the GPU, want 1-3 GB", all.WeightsMB)
	}
	if !hasNote(all, "experts of 40 block(s)") {
		t.Errorf("expected 40 expert blocks on host: %v", all.Notes)
	}
	// KV is unaffected: the blocks themselves stay on the GPU.
	if all.KVMB != full.KVMB {
		t.Errorf("kv changed with --cpu-moe: %d vs %d", all.KVMB, full.KVMB)
	}
	// --n-cpu-moe 20: half the expert bytes.
	if lo, hi := all.HostWeightsMB*45/100, all.HostWeightsMB*55/100; half.HostWeightsMB < lo || half.HostWeightsMB > hi {
		t.Errorf("--n-cpu-moe 20 host weights = %d, want ~half of %d", half.HostWeightsMB, all.HostWeightsMB)
	}
	if !hasNote(half, "experts of 20 block(s)") {
		t.Errorf("expected 20 expert blocks on host: %v", half.Notes)
	}
	// The equivalent -ot rule gives the same answer.
	if ot.WeightsMB != all.WeightsMB || ot.HostWeightsMB != all.HostWeightsMB {
		t.Errorf("-ot exps=CPU should match --cpu-moe: %s vs %s", ot, all)
	}
}
