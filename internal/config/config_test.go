package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/job"
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
	if c.GPU.Device != "auto" || c.GPU.HeadroomMB != DefaultHeadroomMB("auto") || c.GPU.PollInterval != 500*time.Millisecond {
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

func TestBackgroundShare(t *testing.T) {
	c, _ := Parse(nil)
	if c.Policy.BackgroundShareOrDefault() != 0.5 {
		t.Errorf("default share = %v, want 0.5", c.Policy.BackgroundShareOrDefault())
	}
	c, err := Parse([]byte("policy:\n  background_share: 0.25\n"))
	if err != nil || c.Policy.BackgroundShareOrDefault() != 0.25 {
		t.Errorf("share 0.25: %v %v", c, err)
	}
	for _, bad := range []string{"0", "-0.1", "1.5"} {
		if _, err := Parse([]byte("policy:\n  background_share: " + bad + "\n")); err == nil {
			t.Errorf("background_share %s should be rejected", bad)
		}
	}
}

func TestEvictionPolicy(t *testing.T) {
	c, err := Parse(nil)
	if err != nil {
		t.Fatal(err)
	}
	if c.Policy.Eviction != EvictionCost || c.Policy.MinResidencyOrDefault() != 30*time.Second {
		t.Errorf("defaults: eviction=%q min_residency=%v", c.Policy.Eviction, c.Policy.MinResidencyOrDefault())
	}
	c, err = Parse([]byte("policy:\n  eviction: lru\n  min_residency: 0s\n"))
	if err != nil {
		t.Fatal(err)
	}
	if c.Policy.Eviction != EvictionLRU || c.Policy.MinResidencyOrDefault() != 0 {
		t.Errorf("explicit: eviction=%q min_residency=%v", c.Policy.Eviction, c.Policy.MinResidencyOrDefault())
	}
	if c.Policy.BatchMaxStarvationOrDefault() != 2*time.Minute {
		t.Errorf("batch_max_starvation default = %v", c.Policy.BatchMaxStarvationOrDefault())
	}
	if c, _ := Parse([]byte("policy:\n  batch_max_starvation: 0s\n")); c.Policy.BatchMaxStarvationOrDefault() != 0 {
		t.Error("0s must disable the bound")
	}
	for _, bad := range []string{"policy:\n  eviction: lfu\n", "policy:\n  min_residency: -1s\n", "policy:\n  batch_max_starvation: -1s\n"} {
		if _, err := Parse([]byte(bad)); err == nil {
			t.Errorf("%q should be rejected", bad)
		}
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

func TestAppleDeviceHeadroom(t *testing.T) {
	c, err := Parse([]byte(minimal + "gpu:\n  device: apple\n"))
	if err != nil {
		t.Fatal(err)
	}
	if c.GPU.HeadroomMB != 1024 {
		t.Errorf("apple headroom = %d, want 1024", c.GPU.HeadroomMB)
	}
	c, err = Parse([]byte(minimal + "gpu:\n  device: apple\n  headroom_mb: 2048\n"))
	if err != nil || c.GPU.HeadroomMB != 2048 {
		t.Errorf("explicit headroom must win: %v %v", c.GPU.HeadroomMB, err)
	}
	if DefaultHeadroomMB("nvidia:0") != 512 || DefaultHeadroomMB("simulation") != 512 {
		t.Error("dedicated devices keep 512")
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

func TestSimulationRuntimeNeedsNoPath(t *testing.T) {
	src := `
runtimes:
  sim: {type: simulation}
models:
  m: {runtime: sim, capabilities: [chat], simulated_vram_mb: 9000, simulated_load_time: 3s}
`
	c, err := Parse([]byte(src))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.Models["m"].SimulatedVRAMMB != 9000 || c.Models["m"].SimulatedLoad != 3*time.Second {
		t.Errorf("simulation knobs = %+v", c.Models["m"])
	}
	if err := c.CheckFiles(); err != nil {
		t.Errorf("CheckFiles should skip simulation runtimes: %v", err)
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

const familyModels = `
runtimes:
  sim: {type: simulation}
models:
  big:   {runtime: sim, capabilities: [chat, vision], simulated_vram_mb: 8000, aliases: [gpt-4o]}
  mid:   {runtime: sim, capabilities: [chat, vision], simulated_vram_mb: 4000}
  small: {runtime: sim, capabilities: [chat], simulated_vram_mb: 3000}
  embed: {runtime: sim, capabilities: [embedding], simulated_vram_mb: 400}
`

func TestFamilies(t *testing.T) {
	c, err := Parse([]byte(familyModels + `
families:
  gemma: {preferred: big, balanced: mid, compact: small, aliases: [default-chat]}
  two:   {preferred: big, compact: small}
`))
	if err != nil {
		t.Fatal(err)
	}
	f := c.Families["gemma"]
	for q, want := range map[Quality]string{
		QualityPreferred: "big", QualityBalanced: "big,mid", QualityCompact: "big,mid,small",
	} {
		if got := strings.Join(f.Variants(q), ","); got != want {
			t.Errorf("Variants(%s) = %s, want %s", q, got, want)
		}
	}
	if got := strings.Join(c.Families["two"].Variants(QualityBalanced), ","); got != "big" {
		t.Errorf("a missing tier is skipped: balanced of {big, small} = %s, want big", got)
	}
	if id, ok := c.ResolveFamily("default-chat"); !ok || id != "gemma" {
		t.Errorf("alias resolution = %q %v", id, ok)
	}
	if _, ok := c.ResolveFamily("big"); ok {
		t.Error("a model id is not a family")
	}
	if q, err := ParseQuality(""); err != nil || q != QualityCompact {
		t.Errorf(`ParseQuality("") = %q %v`, q, err)
	}
	if _, err := ParseQuality("best"); err == nil {
		t.Error("unknown quality must be rejected")
	}
}

func TestFamilyValidation(t *testing.T) {
	cases := map[string]string{
		"empty":                    "  f: {}\n",
		"unknown variant":          "  f: {preferred: nope}\n",
		"embedding":                "  f: {preferred: embed}\n",
		"twice":                    "  f: {preferred: big, compact: big}\n",
		"clashes model":            "  big: {preferred: mid}\n",
		"clashes alias":            "  f: {preferred: mid, aliases: [gpt-4o]}\n",
		"at in id":                 "  f@x: {preferred: mid}\n",
		"two families, same alias": "  f: {preferred: mid, aliases: [x]}\n  g: {preferred: small, aliases: [x]}\n",
	}
	for name, fam := range cases {
		if _, err := Parse([]byte(familyModels + "families:\n" + fam)); err == nil {
			t.Errorf("%s: should be rejected", name)
		}
	}
}
