package scheduler

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gridcore/gridcore/internal/config"
	gpufake "github.com/gridcore/gridcore/internal/gpu/fake"
	"github.com/gridcore/gridcore/internal/job"
	"github.com/gridcore/gridcore/internal/metrics"
	"github.com/gridcore/gridcore/internal/model"
	"github.com/gridcore/gridcore/internal/runtime"
	rtfake "github.com/gridcore/gridcore/internal/runtime/fake"
)

// fakeClock lets tests move policy time (hot_ttl, idle window, deadlines)
// without sleeping. Wake-ups still come from the real ticker.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

var testPortBase atomic.Int64

func init() { testPortBase.Store(43000) }

// harness runs a scheduler against the fake runtime and fake GPU.
type harness struct {
	t      *testing.T
	cfg    *config.Config
	gpu    *gpufake.Monitor
	rt     *rtfake.Runtime
	s      *Scheduler
	clock  *fakeClock
	cancel context.CancelFunc
	done   chan struct{}
	nextID atomic.Int64
}

// modelsYAML builds the models section. Each model: id -> "vram[,parallel[,flags]]".
const baseYAML = `
gpu: { device: fake, headroom_mb: 512, poll_interval: 10ms }
runtimes:
  sim: { type: fake, port_range: [%d, %d] }
policy:
  classes:
    interactive: { hot_ttl: 200ms }
  interactive_idle_before_background: 30ms
  embedding_chunk_size: 4
models:
%s
`

func newHarness(t *testing.T, totalMB int, models string) *harness {
	t.Helper()
	return newHarnessWith(t, totalMB, models, "")
}

// newHarnessWith appends extra lines to the policy section.
func newHarnessWith(t *testing.T, totalMB int, models, extraPolicy string) *harness {
	t.Helper()
	lo := int(testPortBase.Add(50))
	src := fmt.Sprintf(baseYAML, lo, lo+49, models)
	if extraPolicy != "" {
		src = strings.Replace(src, "  embedding_chunk_size: 4\n", "  embedding_chunk_size: 4\n"+extraPolicy, 1)
	}
	cfg, err := config.Parse([]byte(src))
	if err != nil {
		t.Fatalf("config: %v\n%s", err, src)
	}
	gpu := gpufake.New("FakeGPU", totalMB)
	rt := rtfake.New("sim", gpu)
	rt.RequestDelay = time.Millisecond
	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}

	store, _ := model.OpenStore("")
	s, err := New(cfg, map[string]runtime.Runtime{"sim": rt}, gpu, store, metrics.New(), Options{
		Tick:            5 * time.Millisecond,
		Now:             clock.Now,
		Logger:          slog.New(slog.NewTextHandler(testLogWriter{t}, &slog.HandlerOptions{Level: slog.LevelDebug})),
		BreakerWindow:   time.Minute,
		ShutdownTimeout: 2 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	h := &harness{t: t, cfg: cfg, gpu: gpu, rt: rt, s: s, clock: clock, cancel: cancel, done: make(chan struct{})}
	go func() {
		_ = s.Run(ctx)
		close(h.done)
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case <-h.done:
		case <-time.After(5 * time.Second):
			t.Error("scheduler did not shut down")
		}
	})
	return h
}

func (h *harness) submit(c job.Class, modelID string, steps int, opts ...func(*job.Job)) *Handle {
	h.t.Helper()
	id := fmt.Sprintf("%s-%d", c, h.nextID.Add(1))
	j := job.New(id, context.Background(), c, job.Chat, modelID)
	j.Steps = steps
	j.Enqueued = h.clock.Now()
	for _, o := range opts {
		o(j)
	}
	hd, err := h.s.Submit(j)
	if err != nil {
		h.t.Fatalf("submit %s: %v", id, err)
	}
	return hd
}

func withCtx(ctx context.Context) func(*job.Job) { return func(j *job.Job) { j.Ctx = ctx } }
func withMaxWait(d time.Duration) func(*job.Job) { return func(j *job.Job) { j.MaxWait = d } }

// grant waits for the next grant on hd.
func (h *harness) grant(hd *Handle, within time.Duration) Grant {
	h.t.Helper()
	select {
	case g := <-hd.Grants():
		return g
	case <-hd.Done():
		h.t.Fatalf("%s finished before grant: %v", hd.job.ID, hd.Err())
	case <-time.After(within):
		h.t.Fatalf("%s: no grant within %s; state=%s", hd.job.ID, within, h.dump())
	}
	return Grant{}
}

// noGrant asserts hd receives no grant for d.
func (h *harness) noGrant(hd *Handle, d time.Duration) {
	h.t.Helper()
	select {
	case g := <-hd.Grants():
		h.t.Fatalf("%s: unexpected grant step %d; state=%s", hd.job.ID, g.Step, h.dump())
	case <-hd.Done():
		h.t.Fatalf("%s: unexpectedly finished: %v", hd.job.ID, hd.Err())
	case <-time.After(d):
	}
}

// finish waits for the job to be terminal and returns its error.
func (h *harness) finish(hd *Handle, within time.Duration) error {
	h.t.Helper()
	select {
	case <-hd.Done():
		return hd.Err()
	case <-time.After(within):
		h.t.Fatalf("%s: not finished within %s; state=%s", hd.job.ID, within, h.dump())
	}
	return nil
}

// runToCompletion drives a job: every grant is completed immediately.
func (h *harness) runToCompletion(hd *Handle, within time.Duration) error {
	h.t.Helper()
	deadline := time.After(within)
	for {
		select {
		case g := <-hd.Grants():
			hd.StepDone(g.Step, nil, Usage{})
		case <-hd.Done():
			return hd.Err()
		case <-deadline:
			h.t.Fatalf("%s: not complete within %s; state=%s", hd.job.ID, within, h.dump())
		}
	}
}

func (h *harness) eventually(cond func(st State) bool, msg string) {
	h.t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond(h.s.State()) {
			return
		}
		time.Sleep(3 * time.Millisecond)
	}
	h.t.Fatalf("condition not met: %s\nstate=%s", msg, h.dump())
}

func (h *harness) resident(id string) *ResidentModel {
	st := h.s.State()
	for i := range st.Resident {
		if st.Resident[i].ID == id {
			return &st.Resident[i]
		}
	}
	return nil
}

func isReady(id string) func(State) bool {
	return func(st State) bool {
		for _, r := range st.Resident {
			if r.ID == id && r.State == "ready" {
				return true
			}
		}
		return false
	}
}

func notResident(id string) func(State) bool {
	return func(st State) bool {
		for _, r := range st.Resident {
			if r.ID == id {
				return false
			}
		}
		return true
	}
}

func (h *harness) events(kind string) []Event {
	var out []Event
	for _, e := range h.s.State().Events {
		if e.Kind == kind {
			out = append(out, e)
		}
	}
	return out
}

func (h *harness) dump() string {
	st := h.s.State()
	var b strings.Builder
	fmt.Fprintf(&b, "\n mode=%s budget=%d committed=%d avail=%d\n", st.Mode, st.GPU.BudgetMB, st.GPU.CommittedMB, st.GPU.AvailableMB)
	for _, r := range st.Resident {
		fmt.Fprintf(&b, " resident %s %s %s vram=%d busy=%d/%d\n", r.ID, r.State, r.Tier, r.VRAMMB, r.BusySlots, r.Slots)
	}
	for _, j := range st.Running {
		fmt.Fprintf(&b, " running %s %s %s %d/%d/%d\n", j.ID, j.Class, j.Model, j.Completed, j.Dispatched, j.Steps)
	}
	for _, j := range st.Queued {
		fmt.Fprintf(&b, " queued %s %s %s waiting_for=%q\n", j.ID, j.Class, j.Model, j.Reason)
	}
	n := len(st.Events)
	if n > 12 {
		st.Events = st.Events[n-12:]
	}
	for _, e := range st.Events {
		fmt.Fprintf(&b, " ev %-9s %-14s %s\n", e.Kind, e.Subject, e.Detail)
	}
	return b.String()
}

type testLogWriter struct{ t *testing.T }

func (w testLogWriter) Write(p []byte) (int, error) {
	if os.Getenv("GRIDCORE_TEST_LOG") != "" {
		w.t.Log(strings.TrimSpace(string(p)))
	}
	return len(p), nil
}
