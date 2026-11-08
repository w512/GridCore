package scheduler

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/config"
	gpusim "github.com/w512/gridcore/internal/gpu/simulation"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/runtime"
	rtsim "github.com/w512/gridcore/internal/runtime/simulation"
)

// Neither runtime binds until both loads have started. A TCP probe alone
// therefore cannot see the first load's port reservation.
func TestOverlappingRuntimePorts(t *testing.T) {
	lo := int(testPortBase.Add(20))
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
gpu: {device: simulation, headroom_mb: 512, poll_interval: 10ms}
runtimes:
  first: {type: simulation, port_range: [%d, %d]}
  second: {type: simulation, port_range: [%d, %d]}
models:
  a: {runtime: first, capabilities: [chat], simulated_vram_mb: 1000}
  b: {runtime: second, capabilities: [chat], simulated_vram_mb: 1000}
`, lo, lo+2, lo, lo+2)))
	if err != nil {
		t.Fatal(err)
	}
	mon := gpusim.New("Simulated GPU", 16000)
	started := make(chan int, 2)
	release := make(chan struct{})
	releaseLoads := sync.OnceFunc(func() { close(release) })
	runtimes := map[string]runtime.Runtime{}
	for _, name := range []string{"first", "second"} {
		rt := rtsim.New(name, mon)
		runtimes[name] = &shutdownRuntime{Runtime: rt, load: func(ctx context.Context, sp *model.Spec, port int) (runtime.Instance, error) {
			started <- port
			select {
			case <-release:
				return rt.Load(ctx, sp, port)
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}}
	}
	s, err := New(cfg, runtimes, mon, nil, nil, Options{
		Tick: 5 * time.Millisecond, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = s.Run(ctx); close(done) }()
	t.Cleanup(func() {
		cancel()
		releaseLoads()
		select {
		case <-done:
		case <-time.After(wait):
			t.Error("scheduler did not shut down")
		}
	})
	var ports []int
	for _, id := range []string{"a", "b"} {
		if err := s.LoadModel(id); err != nil {
			t.Fatal(err)
		}
		select {
		case port := <-started:
			ports = append(ports, port)
		case <-time.After(wait):
			t.Fatal("load did not start")
		}
	}
	if ports[0] == ports[1] {
		t.Fatalf("different runtimes reserved the same port: %d", ports[0])
	}
	releaseLoads()
	h := &harness{t: t, s: s}
	h.eventually(func(st State) bool { return isReady("a")(st) && isReady("b")(st) }, "both runtimes ready")
}
