package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gridcore/gridcore/internal/job"
)

const minimal = `
runtimes:
  llamacpp:
    binary: /usr/bin/true
models:
  chat:
    runtime: llamacpp
    path: /models/chat.gguf
    capabilities: [chat]
    aliases: [gpt-4o]
  embed:
    runtime: llamacpp
    path: /models/embed.gguf
    capabilities: [embedding]
    pinned: true
`

func TestParseAppliesDefaults(t *testing.T) {
	c, err := Parse([]byte(minimal))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.Server.Listen != "127.0.0.1:8080" {
		t.Errorf("listen default = %q", c.Server.Listen)
	}
	if c.GPU.Device != "auto" || c.GPU.HeadroomMB != 512 || c.GPU.PollInterval != 500*time.Millisecond {
		t.Errorf("gpu defaults = %+v", c.GPU)
	}
	rt := c.Runtimes["llamacpp"]
	if rt.Type != RuntimeLlamaCpp || rt.PortRange != [2]int{41000, 41999} || rt.LoadTimeout != 120*time.Second {
		t.Errorf("runtime defaults = %+v", rt)
	}
	m := c.Models["chat"]
	if m.Ctx != 4096 || m.Parallel != 1 {
		t.Errorf("model defaults = %+v", m)
	}
	if !c.Models["embed"].Preload {
		t.Error("pinned should imply preload")
	}
	if c.Policy.DefaultClass != job.Interactive {
		t.Errorf("default_class = %q", c.Policy.DefaultClass)
	}
	if p := c.Policy.Classes[job.Interactive]; p.Priority != 100 || p.HotTTL != 5*time.Minute {
		t.Errorf("interactive policy = %+v", p)
	}
	if c.Policy.Classes[job.Background].Priority != 30 || c.Policy.Classes[job.Batch].Priority != 10 {
		t.Errorf("class priorities = %+v", c.Policy.Classes)
	}
	if c.Policy.EmbeddingChunkSize != 32 || c.Policy.InteractiveIdleBeforeBackground != 2*time.Second {
		t.Errorf("policy defaults = %+v", c.Policy)
	}
	if c.Policy.MaxStarvation() != 3*time.Second {
		t.Errorf("background_max_starvation default = %v", c.Policy.MaxStarvation())
	}
}

func TestStarvationZeroDisables(t *testing.T) {
	c, err := Parse([]byte("policy:\n  background_max_starvation: 0s\n"))
	if err != nil {
		t.Fatal(err)
	}
	if c.Policy.MaxStarvation() != 0 {
		t.Errorf("0s must disable, got %v", c.Policy.MaxStarvation())
	}
	c, _ = Parse([]byte("policy:\n  background_max_starvation: 500ms\n"))
	if c.Policy.MaxStarvation() != 500*time.Millisecond {
		t.Errorf("got %v", c.Policy.MaxStarvation())
	}
}

func TestParseEmptyIsValid(t *testing.T) {
	c, err := Parse(nil)
	if err != nil {
		t.Fatalf("empty config should be valid: %v", err)
	}
	if len(c.Models) != 0 || len(c.Runtimes) != 0 {
		t.Errorf("expected no models/runtimes, got %+v", c)
	}
}

func TestParseRejectsUnknownKeys(t *testing.T) {
	_, err := Parse([]byte("server:\n  listne: 127.0.0.1:1\n"))
	if err == nil || !strings.Contains(err.Error(), "listne") {
		t.Fatalf("expected unknown-field error, got %v", err)
	}
}

func TestParseDurationsAndOverrides(t *testing.T) {
	src := minimal + `
gpu:
  device: nvidia:1
  vram_limit_mb: 12000
  headroom_mb: 256
policy:
  default_class: background
  classes:
    interactive: { hot_ttl: 30s }
  embedding_chunk_size: 8
`
	c, err := Parse([]byte(src))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.GPU.VRAMLimitMB == nil || *c.GPU.VRAMLimitMB != 12000 {
		t.Errorf("vram_limit_mb = %v", c.GPU.VRAMLimitMB)
	}
	if c.Policy.DefaultClass != job.Background {
		t.Errorf("default_class = %q", c.Policy.DefaultClass)
	}
	p := c.Policy.Classes[job.Interactive]
	if p.HotTTL != 30*time.Second {
		t.Errorf("hot_ttl = %v", p.HotTTL)
	}
	if p.Priority != 100 {
		t.Errorf("priority should keep default when only hot_ttl is set, got %d", p.Priority)
	}
	if c.Policy.EmbeddingChunkSize != 8 {
		t.Errorf("chunk = %d", c.Policy.EmbeddingChunkSize)
	}
}

func TestValidateErrors(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want string
	}{
		{
			"unknown runtime",
			"runtimes:\n  llamacpp: {binary: /x}\nmodels:\n  m: {runtime: nope, path: /p, capabilities: [chat]}\n",
			`runtime "nope" is not defined`,
		},
		{
			"missing binary",
			"runtimes:\n  llamacpp: {}\n",
			"binary is required",
		},
		{
			"bad runtime type",
			"runtimes:\n  x: {type: vllm, binary: /x}\n",
			`type "vllm"`,
		},
		{
			"missing capabilities",
			"runtimes:\n  llamacpp: {binary: /x}\nmodels:\n  m: {runtime: llamacpp, path: /p}\n",
			"capabilities must not be empty",
		},
		{
			"unknown capability",
			"runtimes:\n  llamacpp: {binary: /x}\nmodels:\n  m: {runtime: llamacpp, path: /p, capabilities: [ocr]}\n",
			`unknown "ocr"`,
		},
		{
			"alias collides with id",
			"runtimes:\n  llamacpp: {binary: /x}\nmodels:\n  a: {runtime: llamacpp, path: /p, capabilities: [chat], aliases: [b]}\n  b: {runtime: llamacpp, path: /p, capabilities: [chat]}\n",
			"already used",
		},
		{
			"alias collides with alias",
			"runtimes:\n  llamacpp: {binary: /x}\nmodels:\n  a: {runtime: llamacpp, path: /p, capabilities: [chat], aliases: [x]}\n  b: {runtime: llamacpp, path: /p, capabilities: [chat], aliases: [x]}\n",
			`"x" already used`,
		},
		{
			"id with @",
			"runtimes:\n  llamacpp: {binary: /x}\nmodels:\n  a@b: {runtime: llamacpp, path: /p, capabilities: [chat]}\n",
			"must not contain '@'",
		},
		{
			"bad port range",
			"runtimes:\n  llamacpp: {binary: /x, port_range: [5000, 4000]}\n",
			"port_range",
		},
		{
			"bad device",
			"gpu:\n  device: cuda:0\n",
			"gpu.device",
		},
		{
			"bad default class",
			"policy:\n  default_class: urgent\n",
			"default_class",
		},
		{
			"unknown class key",
			"policy:\n  classes:\n    realtime: {priority: 90}\n",
			"unknown class",
		},
		{
			"missing path for real runtime",
			"runtimes:\n  llamacpp: {binary: /x}\nmodels:\n  m: {runtime: llamacpp, capabilities: [chat]}\n",
			"path is required",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Parse([]byte(tc.src))
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}

func TestFakeRuntimeNeedsNoPath(t *testing.T) {
	src := `
runtimes:
  sim: {type: fake}
models:
  m: {runtime: sim, capabilities: [chat], fake_vram_mb: 9000, fake_load_time: 3s}
`
	c, err := Parse([]byte(src))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.Models["m"].FakeVRAMMB != 9000 || c.Models["m"].FakeLoad != 3*time.Second {
		t.Errorf("fake knobs = %+v", c.Models["m"])
	}
	if err := c.CheckFiles(); err != nil {
		t.Errorf("CheckFiles should skip fake runtimes: %v", err)
	}
}

func TestCheckFilesReportsMissing(t *testing.T) {
	c, err := Parse([]byte(minimal))
	if err != nil {
		t.Fatal(err)
	}
	err = c.CheckFiles()
	if err == nil || !strings.Contains(err.Error(), "models.chat.path") {
		t.Fatalf("expected missing model path error, got %v", err)
	}
}

func TestResolveModel(t *testing.T) {
	c, err := Parse([]byte(minimal))
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{"chat": "chat", "gpt-4o": "chat", "embed": "embed"} {
		got, ok := c.ResolveModel(name)
		if !ok || got != want {
			t.Errorf("ResolveModel(%q) = %q,%v want %q", name, got, ok, want)
		}
	}
	if _, ok := c.ResolveModel("nope"); ok {
		t.Error("unknown name should not resolve")
	}
	if !c.Models["chat"].HasCapability(CapChat) || c.Models["chat"].HasCapability(CapEmbedding) {
		t.Error("HasCapability mismatch")
	}
}

func TestExpandHome(t *testing.T) {
	home, _ := os.UserHomeDir()
	c, err := Parse([]byte(strings.ReplaceAll(minimal, "/models/chat.gguf", "~/m/chat.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := c.Models["chat"].Path, filepath.Join(home, "m", "chat.gguf"); got != want {
		t.Errorf("path = %q want %q", got, want)
	}
}

func TestExampleConfigIsValid(t *testing.T) {
	c, err := Load(filepath.Join("..", "..", "config.example.yaml"))
	if err != nil {
		t.Fatalf("config.example.yaml must parse and validate: %v", err)
	}
	if len(c.Models) != 3 {
		t.Errorf("example should define 3 models, got %d", len(c.Models))
	}
}
