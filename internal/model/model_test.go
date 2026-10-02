package model

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/config"
)

func TestFromConfig(t *testing.T) {
	cfg, err := config.Parse([]byte(`
runtimes:
  llamacpp: {binary: /x}
  sim: {type: fake}
models:
  chat: {runtime: llamacpp, path: /m.gguf, capabilities: [chat, vision], ctx: 8192, parallel: 2, args: ["--jinja"], aliases: [gpt-4o]}
  sim1: {runtime: sim, capabilities: [embedding], pinned: true, fake_vram_mb: 700}
`))
	if err != nil {
		t.Fatal(err)
	}
	specs := FromConfig(cfg)
	c := specs["chat"]
	if c.RuntimeType != config.RuntimeLlamaCpp || c.Ctx != 8192 || c.Parallel != 2 || !c.HasCapability("vision") {
		t.Errorf("chat spec = %+v", c)
	}
	s := specs["sim1"]
	if s.RuntimeType != config.RuntimeFake || !s.Pinned || !s.Preload || s.EstimateVRAMMB() != 700 {
		t.Errorf("sim spec = %+v", s)
	}
}

func TestProfileKeyIgnoresArgOrderButNotValues(t *testing.T) {
	a := &Spec{RuntimeType: "llamacpp", Path: "/m", Ctx: 4096, Parallel: 1, Args: []string{"--a", "--b"}}
	b := &Spec{RuntimeType: "llamacpp", Path: "/m", Ctx: 4096, Parallel: 1, Args: []string{"--b", "--a"}}
	if a.ProfileKey("bin", "gpu") != b.ProfileKey("bin", "gpu") {
		t.Error("arg order should not change the key")
	}
	c := &Spec{RuntimeType: "llamacpp", Path: "/m", Ctx: 8192, Parallel: 1, Args: []string{"--a", "--b"}}
	if a.ProfileKey("bin", "gpu") == c.ProfileKey("bin", "gpu") {
		t.Error("ctx must change the key")
	}
	if a.ProfileKey("bin", "gpu") == a.ProfileKey("bin2", "gpu") {
		t.Error("binary must change the key")
	}
	if a.ProfileKey("bin", "gpu") == a.ProfileKey("bin", "gpu2") {
		t.Error("gpu must change the key")
	}
}

func TestProfileKeyFakeModelsDiffer(t *testing.T) {
	a := &Spec{ID: "a", RuntimeType: "fake", Ctx: 4096, Parallel: 1}
	b := &Spec{ID: "b", RuntimeType: "fake", Ctx: 4096, Parallel: 1}
	if a.ProfileKey("fake", "gpu") == b.ProfileKey("fake", "gpu") {
		t.Error("fake models without a path must not share a profile")
	}
}

func TestEstimateVRAMUnknownFile(t *testing.T) {
	s := &Spec{RuntimeType: "llamacpp", Path: "/does/not/exist.gguf", Ctx: 4096, Parallel: 1}
	if got := s.EstimateVRAMMB(); got != 0 {
		t.Errorf("unknown file should estimate 0, got %d", got)
	}
}

func TestProfileObservations(t *testing.T) {
	var p Profile
	p.ObserveLoad(10 * time.Second)
	if p.LoadMS != 10000 || p.Samples != 1 {
		t.Errorf("first load = %+v", p)
	}
	p.ObserveLoad(5 * time.Second)
	if p.LoadMS >= 10000 || p.LoadMS <= 5000 {
		t.Errorf("EMA should move toward 5000, got %v", p.LoadMS)
	}
	p.ObserveVRAM(9000)
	p.ObserveVRAM(8500)
	if p.VRAMMB != 9000 {
		t.Errorf("VRAM should keep max, got %d", p.VRAMMB)
	}
	p.ObserveThroughput(0, 20)
	if p.PromptTPS != 0 || p.GenTPS != 20 {
		t.Errorf("throughput = %+v", p)
	}
}

func TestStoreRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state", "profiles.json")
	s, err := OpenStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Update("k1", "chat", func(p *Profile) { p.ObserveVRAM(9100); p.ObserveLoad(3 * time.Second) }); err != nil {
		t.Fatal(err)
	}
	s2, err := OpenStore(path)
	if err != nil {
		t.Fatal(err)
	}
	p, ok := s2.Get("k1")
	if !ok || p.VRAMMB != 9100 || p.LoadMS != 3000 || p.ModelID != "chat" {
		t.Errorf("reloaded profile = %+v ok=%v", p, ok)
	}
	if _, ok := s2.Get("missing"); ok {
		t.Error("missing key should not be found")
	}
	if len(s2.All()) != 1 {
		t.Errorf("All() = %d", len(s2.All()))
	}
}

func TestStoreSkipsUnchangedWrites(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profiles.json")
	s, _ := OpenStore(path)
	if err := s.Update("k", "m", func(p *Profile) { p.ObserveVRAM(9000) }); err != nil {
		t.Fatal(err)
	}
	before := statFile(path)
	time.Sleep(10 * time.Millisecond) // a rewrite would get a new mtime
	// The scheduler reports every GPU snapshot; most change nothing.
	if err := s.Update("k", "m", func(p *Profile) { p.ObserveVRAM(8000) }); err != nil {
		t.Fatal(err)
	}
	if after := statFile(path); after != before {
		t.Error("an update that changes nothing must not rewrite the file")
	}
	if err := s.Update("k", "m", func(p *Profile) { p.ObserveVRAM(9500) }); err != nil {
		t.Fatal(err)
	}
	if after := statFile(path); after == before {
		t.Error("a real change must be written")
	}
}

// The daemon keeps its store open for hours while `gridcore bench` and
// `gridcore profiles prune` change the same file.
func TestStoreMergesOtherWriters(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profiles.json")
	daemon, _ := OpenStore(path)
	_ = daemon.Update("chat", "chat", func(p *Profile) { p.ObserveVRAM(8000) })
	_ = daemon.Update("old", "old", func(p *Profile) { p.ObserveVRAM(100) })

	bench, _ := OpenStore(path)
	_ = bench.Update("embed", "embed", func(p *Profile) { p.ObserveVRAM(400) })
	_ = bench.Update("chat", "chat", func(p *Profile) { p.ObserveVRAM(8500) }) // newer measurement

	prune, _ := OpenStore(path)
	if err := prune.Delete("old"); err != nil {
		t.Fatal(err)
	}

	// The daemon's next write must keep bench's entries, take the newer
	// value for "chat" and not resurrect the pruned entry.
	_ = daemon.Update("vision", "vision", func(p *Profile) { p.ObserveVRAM(6000) })

	final, _ := OpenStore(path)
	got := map[string]int{}
	for _, p := range final.All() {
		got[p.Key] = p.VRAMMB
	}
	want := map[string]int{"chat": 8500, "embed": 400, "vision": 6000}
	if len(got) != len(want) {
		t.Fatalf("profiles = %v, want %v", got, want)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s = %d, want %d (all: %v)", k, got[k], v, got)
		}
	}
	if p, ok := daemon.Get("chat"); !ok || p.VRAMMB != 8500 {
		t.Errorf("daemon should have adopted the newer measurement, got %+v", p)
	}
}

func TestStoreInMemory(t *testing.T) {
	s, err := OpenStore("")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Update("k", "m", func(p *Profile) { p.ObserveVRAM(1) }); err != nil {
		t.Fatalf("in-memory store should not fail to save: %v", err)
	}
}
