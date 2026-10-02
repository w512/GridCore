package scheduler

import (
	"context"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/config"
	gpusim "github.com/w512/gridcore/internal/gpu/simulation"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/runtime"
	"github.com/w512/gridcore/internal/runtime/llamacpp"
	rtsim "github.com/w512/gridcore/internal/runtime/simulation"
)

// Profiles survive reinstalling the same llama.cpp build (new mtime) and
// start fresh for a different build.
func TestRuntimeIDFollowsBuildNotFile(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "llama-server")
	write := func(build string) config.Runtime {
		t.Helper()
		stub := "#!/bin/sh\necho 'version: " + build + " (a1b2c3d)' >&2\n"
		if err := os.WriteFile(bin, []byte(stub), 0o755); err != nil {
			t.Fatal(err)
		}
		return config.Runtime{Type: config.RuntimeLlamaCpp, Binary: bin}
	}
	id := func(rc config.Runtime) string { return RuntimeID(rc, llamacpp.New("llamacpp", rc, "")) }

	rc := write("11060")
	first := id(rc)
	if first != "b11060-a1b2c3d" {
		t.Fatalf("RuntimeID = %q", first)
	}
	later := time.Now().Add(time.Hour)
	if err := os.Chtimes(bin, later, later); err != nil {
		t.Fatal(err)
	}
	if again := id(rc); again != first {
		t.Errorf("same build, new mtime: %q != %q", again, first)
	}
	if other := id(write("11100")); other == first {
		t.Errorf("a different build must change the id, got %q", other)
	}

	// A binary that does not report a version falls back to path+mtime+size.
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if fb := id(rc); fb == "" || fb == first || filepath.Dir(fb) != filepath.Dir(bin) {
		t.Errorf("fallback id = %q", fb)
	}
}

type versionedSim struct {
	*rtsim.Runtime
	v string
}

func (r versionedSim) Version() string { return r.v }

// A profile recorded by 0.1 (keyed by binary path/mtime/size) for the binary
// on disk now is carried over to the build-id key instead of being lost.
func TestLegacyProfileCarriedOver(t *testing.T) {
	cfg, err := config.Parse([]byte(`
gpu: { device: simulation, poll_interval: 10ms }
runtimes:
  sim: { type: simulation, port_range: [44900, 44949] }
models:
  chat: { runtime: sim, capabilities: [chat], simulated_vram_mb: 9000 }
`))
	if err != nil {
		t.Fatal(err)
	}
	sp := model.FromConfig(cfg)["chat"]
	store, _ := model.OpenStore("")
	_ = store.Update(sp.ProfileKey("simulation", "Simulated GPU"), "chat", func(p *model.Profile) {
		p.ObserveVRAM(8800)
		p.ObserveLoad(2 * time.Second)
	})

	gpu := gpusim.New("Simulated GPU", 16000)
	rt := versionedSim{rtsim.New("sim", gpu), "b11060-a1b2c3d"}
	s, err := New(cfg, map[string]runtime.Runtime{"sim": rt}, gpu, store, nil,
		Options{Logger: slog.New(slog.NewTextHandler(io.Discard, nil))})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = s.Run(ctx); close(done) }()
	defer func() { cancel(); <-done }()

	key := sp.ProfileKey("b11060-a1b2c3d", "Simulated GPU")
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if p, ok := store.Get(key); ok {
			if p.VRAMMB != 8800 || p.LoadMS != 2000 || p.Runtime != "b11060-a1b2c3d" || p.GPU != "Simulated GPU" {
				t.Fatalf("carried-over profile = %+v", p)
			}
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("legacy profile was not carried over to the build-id key")
}
