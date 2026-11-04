package scheduler

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/config"
	gpusim "github.com/w512/gridcore/internal/gpu/simulation"
	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/runtime"
	rtsim "github.com/w512/gridcore/internal/runtime/simulation"
)

type shutdownRuntime struct {
	*rtsim.Runtime
	load func(context.Context, *model.Spec, int) (runtime.Instance, error)
}

func (r *shutdownRuntime) Load(ctx context.Context, sp *model.Spec, port int) (runtime.Instance, error) {
	return r.load(ctx, sp, port)
}

func newShutdownScheduler(t *testing.T, rt runtime.Runtime, timeout time.Duration) (*Scheduler, context.CancelFunc, <-chan struct{}) {
	t.Helper()
	lo := int(testPortBase.Add(20))
	cfg, err := config.Parse([]byte(fmt.Sprintf(baseYAML, lo, lo+19,
		"  a: {runtime: sim, capabilities: [chat], simulated_vram_mb: 1000}\n"+
			"  b: {runtime: sim, capabilities: [chat], simulated_vram_mb: 1000}\n")))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(cfg, map[string]runtime.Runtime{"sim": rt}, gpusim.New("Simulated GPU", 16000), nil, nil, Options{
		Tick:            5 * time.Millisecond,
		ShutdownTimeout: timeout,
		Logger:          slog.New(slog.NewTextHandler(io.Discard, nil)),
		ProcessName:     func(int) string { return "test-runtime" },
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = s.Run(ctx); close(done) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(wait):
			t.Error("scheduler did not shut down")
		}
	})
	return s, cancel, done
}

func TestShutdownCancelsLoadsAndWaitsForCleanup(t *testing.T) {
	started := make(chan context.Context, 2)
	release := make(chan struct{})
	releaseLoads := sync.OnceFunc(func() { close(release) })
	defer releaseLoads()
	rt := &shutdownRuntime{Runtime: rtsim.New("sim", nil)}
	rt.load = func(ctx context.Context, _ *model.Spec, _ int) (runtime.Instance, error) {
		started <- ctx
		select {
		case <-ctx.Done():
		case <-release:
		}
		<-release // model cleanup must finish before Run returns
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return nil, errors.New("test released load before cancellation")
	}
	s, cancel, done := newShutdownScheduler(t, rt, 2*time.Second)
	for _, id := range []string{"a", "b"} {
		if err := s.LoadModel(id); err != nil {
			t.Fatal(err)
		}
	}
	var loads []context.Context
	for range 2 {
		select {
		case ctx := <-started:
			loads = append(loads, ctx)
		case <-time.After(wait):
			t.Fatal("load did not start")
		}
	}
	cancel()
	for _, ctx := range loads {
		select {
		case <-ctx.Done():
		case <-time.After(wait):
			t.Fatal("shutdown did not cancel load")
		}
	}
	select {
	case <-done:
		t.Error("Run returned before cancelled loads finished cleanup")
	default:
	}
	if err := s.LoadModel("a"); !errors.Is(err, ErrShuttingDown) {
		t.Errorf("admin load during shutdown: %v", err)
	}
	releaseLoads()
	select {
	case <-done:
	case <-time.After(wait):
		t.Fatal("shutdown did not finish after load cleanup")
	}
	if s.res.Len() != 0 || len(s.ports["sim"].inUse) != 0 {
		t.Error("cancelled loads left residency entries or reserved ports")
	}
	if len(s.breakers) != 0 {
		t.Error("shutdown cancellation counted as a runtime failure")
	}
}

func TestShutdownStopsLateLoadResult(t *testing.T) {
	for _, afterTimeout := range []bool{false, true} {
		name := "during_shutdown"
		timeout := 2 * time.Second
		if afterTimeout {
			name = "after_shutdown_timeout"
			timeout = 20 * time.Millisecond
		}
		t.Run(name, func(t *testing.T) {
			started := make(chan context.Context, 1)
			result := make(chan runtime.Instance, 1)
			loadReturned := make(chan struct{})
			release := make(chan struct{})
			releaseLoad := sync.OnceFunc(func() { close(release) })
			defer releaseLoad()
			rt := &shutdownRuntime{Runtime: rtsim.New("sim", nil)}
			rt.load = func(ctx context.Context, sp *model.Spec, port int) (runtime.Instance, error) {
				defer close(loadReturned)
				started <- ctx
				<-release
				// Readiness can race cancellation; deliberately return a live
				// instance even though the load context was cancelled.
				inst, err := rt.Runtime.Load(context.Background(), sp, port)
				result <- inst
				return inst, err
			}
			s, cancel, done := newShutdownScheduler(t, rt, timeout)
			if err := s.LoadModel("a"); err != nil {
				t.Fatal(err)
			}
			var loadCtx context.Context
			select {
			case loadCtx = <-started:
			case <-time.After(wait):
				t.Fatal("load did not start")
			}
			// Also reap the test instance if the regression leaves it alive.
			t.Cleanup(func() {
				releaseLoad()
				select {
				case <-loadReturned:
				case <-time.After(wait):
					t.Error("test load did not return during cleanup")
				}
				for _, inst := range rt.Instances() {
					_ = inst.Stop(context.Background())
				}
			})
			cancel()
			if afterTimeout {
				select {
				case <-done:
				case <-time.After(wait):
					t.Fatal("shutdown exceeded its timeout waiting for load")
				}
				if !errors.Is(loadCtx.Err(), context.Canceled) {
					t.Errorf("load context after shutdown: %v", loadCtx.Err())
				}
			} else {
				select {
				case <-loadCtx.Done():
				case <-time.After(wait):
					t.Error("shutdown did not cancel load")
				}
				select {
				case <-done:
					t.Error("Run returned while load was still pending")
				default:
				}
			}
			releaseLoad()
			var inst runtime.Instance
			select {
			case inst = <-result:
				if inst == nil {
					t.Fatal("test runtime failed to load instance")
				}
			case <-time.After(wait):
				t.Fatal("load did not return")
			}
			select {
			case <-done:
			case <-time.After(wait):
				t.Fatal("shutdown did not finish")
			}
			if afterTimeout {
				select {
				case <-inst.Done():
				case <-time.After(wait):
					t.Error("late load result left a live instance after shutdown")
				}
			} else {
				select {
				case <-inst.Done():
				default:
					t.Error("Run returned before stopping the loaded instance")
				}
			}
		})
	}
}

type shutdownBlockedStop struct {
	runtime.Instance
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (i *shutdownBlockedStop) Stop(ctx context.Context) error {
	i.once.Do(func() { close(i.started) })
	select {
	case <-i.release:
		return i.Instance.Stop(ctx)
	case <-ctx.Done():
		return ctx.Err()
	}
}

func TestShutdownTimeoutRemainsEffectiveAfterDrain(t *testing.T) {
	stopStarted := make(chan struct{})
	release := make(chan struct{})
	releaseStop := sync.OnceFunc(func() { close(release) })
	defer releaseStop()
	rt := &shutdownRuntime{Runtime: rtsim.New("sim", nil)}
	rt.load = func(ctx context.Context, sp *model.Spec, port int) (runtime.Instance, error) {
		inst, err := rt.Runtime.Load(ctx, sp, port)
		if err != nil {
			return nil, err
		}
		return &shutdownBlockedStop{Instance: inst, started: stopStarted, release: release}, nil
	}
	s, cancel, done := newShutdownScheduler(t, rt, 20*time.Millisecond)
	jobCtx, cancelJob := context.WithCancel(context.Background())
	defer cancelJob()
	h, err := s.Submit(job.New("running", jobCtx, job.Interactive, job.Chat, "a"))
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-h.Grants(): // deliberately leave the step in flight
	case <-time.After(wait):
		t.Fatal("job was not granted")
	}
	cancel()
	select {
	case <-stopStarted:
	case <-time.After(wait):
		t.Fatal("shutdown did not stop instance after drain deadline")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Error("shutdown lost its timeout after the drain phase")
	}
	releaseStop()
	for _, inst := range rt.Instances() {
		select {
		case <-inst.Done():
		case <-time.After(wait):
			t.Error("test instance did not stop")
		}
	}
}
