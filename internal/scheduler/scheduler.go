// Package scheduler owns all decisions: which job runs next, which model is
// loaded or evicted, and how VRAM is reserved.
//
// A single goroutine (Run) owns every piece of mutable state. The API layer
// submits jobs and receives Grants through a Handle; runtimes and the GPU
// monitor report through events. There are no locks around policy code.
package scheduler

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/w512/gridcore/internal/config"
	"github.com/w512/gridcore/internal/gpu"
	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/metrics"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/residency"
	"github.com/w512/gridcore/internal/runtime"
)

// Options tune the loop. Zero values mean defaults.
type Options struct {
	Tick             time.Duration // deadline / idle-window re-evaluation; default 200ms
	Now              func() time.Time
	BreakerThreshold int           // failures within BreakerWindow that disable a model; default 3
	BreakerWindow    time.Duration // default 5m
	ShutdownTimeout  time.Duration // wait for in-flight steps on shutdown; default 30s
	Logger           *slog.Logger
}

func (o *Options) defaults() {
	if o.Tick <= 0 {
		o.Tick = 200 * time.Millisecond
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	if o.BreakerThreshold <= 0 {
		o.BreakerThreshold = 3
	}
	if o.BreakerWindow <= 0 {
		o.BreakerWindow = 5 * time.Minute
	}
	if o.ShutdownTimeout <= 0 {
		o.ShutdownTimeout = 30 * time.Second
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
}

// Scheduler is the control plane for one GPU.
type Scheduler struct {
	cfg      *config.Config
	specs    map[string]*model.Spec
	runtimes map[string]runtime.Runtime
	gpu      gpu.Monitor
	store    *model.Store
	m        *metrics.Metrics
	opts     Options
	log      *slog.Logger

	events  chan event
	stopped chan struct{}

	// ---- loop-owned state below; never touch from other goroutines ----
	q         *queue
	res       *residency.Set
	jobs      map[string]*jobState
	ports     map[string]*portAllocator
	binaryIDs map[string]string
	breakers  map[string]*breaker
	disabled  map[string]string // model id -> reason
	ring      *eventRing

	snap       gpu.Snapshot
	snapOK     bool
	polling    bool
	externalMB int
	// externalSettle is the number of upcoming snapshots that may not raise
	// externalMB. Set when an instance goes away: nvidia-smi answers
	// memory.used and the process list in two calls, so a process that
	// exits between them shows up as unattributed "external" memory for one
	// poll and would trigger a spurious eviction.
	externalSettle int
	gpuName        string

	runningInteractive int
	lastInteractiveEnd time.Time
	mode               string
	shuttingDown       bool

	snapFailures int // consecutive gpu snapshot failures, for log rate-limiting
}

// New wires a scheduler. runtimes is keyed by config runtime name.
func New(cfg *config.Config, runtimes map[string]runtime.Runtime, mon gpu.Monitor, store *model.Store, m *metrics.Metrics, opts Options) (*Scheduler, error) {
	opts.defaults()
	if store == nil {
		var err error
		if store, err = model.OpenStore(""); err != nil {
			return nil, err
		}
	}
	if m == nil {
		m = metrics.New()
	}
	if err := checkPinnedFit(cfg); err != nil {
		return nil, err
	}
	s := &Scheduler{
		cfg:       cfg,
		specs:     model.FromConfig(cfg),
		runtimes:  runtimes,
		gpu:       mon,
		store:     store,
		m:         m,
		opts:      opts,
		log:       opts.Logger.With("component", "scheduler"),
		events:    make(chan event, 256),
		stopped:   make(chan struct{}),
		q:         newQueue(),
		res:       residency.New(cfg.Policy.Classes[job.Interactive].HotTTL),
		jobs:      map[string]*jobState{},
		ports:     map[string]*portAllocator{},
		binaryIDs: map[string]string{},
		breakers:  map[string]*breaker{},
		disabled:  map[string]string{},
		ring:      newEventRing(eventBuffer),
		mode:      "idle",
	}
	for name, rc := range cfg.Runtimes {
		if _, ok := runtimes[name]; !ok {
			return nil, fmt.Errorf("scheduler: no runtime implementation for %q", name)
		}
		s.ports[name] = newPortAllocator(rc.PortRange[0], rc.PortRange[1])
		s.binaryIDs[name] = BinaryID(rc)
	}
	return s, nil
}

// checkPinnedFit rejects a configuration whose pinned models cannot fit in
// an explicitly configured VRAM budget. Pinned models are loaded at start
// and never evicted, so this would otherwise fail on every boot. Without
// vram_limit_mb the device total is unknown until the first snapshot; that
// case is reported at preload time instead.
func checkPinnedFit(cfg *config.Config) error {
	if cfg.GPU.VRAMLimitMB == nil {
		return nil
	}
	budget := *cfg.GPU.VRAMLimitMB - cfg.GPU.HeadroomMB
	specs := model.FromConfig(cfg)
	need := 0
	var ids []string
	for _, id := range sortedSpecIDs(specs) {
		sp := specs[id]
		if !sp.Pinned {
			continue
		}
		est := model.EstimateVRAM(sp).TotalMB
		if est <= 0 {
			continue // unknown file; the load will tell
		}
		need += est + cfg.GPU.HeadroomMB
		ids = append(ids, fmt.Sprintf("%s ~%d MB", id, est))
	}
	if need > budget {
		return fmt.Errorf("pinned models (%s) need about %d MB but gpu.vram_limit_mb %d minus headroom %d leaves %d MB",
			strings.Join(ids, ", "), need, *cfg.GPU.VRAMLimitMB, cfg.GPU.HeadroomMB, budget)
	}
	return nil
}

// BinaryID encodes the runtime binary identity so a rebuilt llama.cpp
// invalidates stored profiles. `gridcore bench` uses it to write profiles
// under the same key the scheduler reads.
func BinaryID(rc config.Runtime) string {
	if rc.Type == config.RuntimeFake {
		return "fake"
	}
	st, err := os.Stat(rc.Binary)
	if err != nil {
		return rc.Binary
	}
	return fmt.Sprintf("%s@%d:%d", rc.Binary, st.ModTime().Unix(), st.Size())
}

// Submit enqueues a job. The returned Handle delivers Grants and the terminal
// result. Submit fails synchronously only for malformed jobs or when the
// scheduler is shutting down.
func (s *Scheduler) Submit(j *job.Job) (*Handle, error) {
	if j.Steps < 1 {
		return nil, errors.New("scheduler: job must have at least one step")
	}
	if _, ok := s.specs[j.ModelID]; !ok {
		return nil, fmt.Errorf("scheduler: unknown model %q", j.ModelID)
	}
	if _, err := job.ParseClass(string(j.Class)); err != nil {
		return nil, err
	}
	if j.Ctx == nil {
		j.Ctx = context.Background()
	}
	if j.Enqueued.IsZero() {
		j.Enqueued = s.opts.Now()
	}
	h := newHandle(s, j)
	js := &jobState{job: j, handle: h, steps: map[int]string{}, stepStart: map[int]time.Time{}}
	if !s.send(evSubmit{js: js}) {
		return nil, ErrShuttingDown
	}
	// Propagate client disconnects.
	go func() {
		select {
		case <-j.Ctx.Done():
			s.send(evCancel{id: j.ID})
		case <-h.done:
		}
	}()
	return h, nil
}

// State returns a consistent snapshot of the scheduler for /admin/state.
func (s *Scheduler) State() State {
	reply := make(chan State, 1)
	if !s.send(evStateReq{reply: reply}) {
		return State{Now: s.opts.Now(), Mode: "stopped"}
	}
	select {
	case st := <-reply:
		return st
	case <-s.stopped:
		return State{Now: s.opts.Now(), Mode: "stopped"}
	}
}

// LoadModel asks the scheduler to make a model resident (admin action). It
// returns once the decision is made, not once the model is loaded.
func (s *Scheduler) LoadModel(id string) error { return s.admin("load", id) }

// UnloadModel evicts a model (admin action). Pinned models cannot be unloaded.
func (s *Scheduler) UnloadModel(id string) error { return s.admin("unload", id) }

// EnableModel clears a circuit breaker.
func (s *Scheduler) EnableModel(id string) error { return s.admin("enable", id) }

func (s *Scheduler) admin(op, id string) error {
	if _, ok := s.specs[id]; !ok {
		return fmt.Errorf("unknown model %q", id)
	}
	reply := make(chan error, 1)
	if !s.send(evAdmin{op: op, model: id, reply: reply}) {
		return ErrShuttingDown
	}
	select {
	case err := <-reply:
		return err
	case <-s.stopped:
		return ErrShuttingDown
	}
}

// send delivers an event to the loop unless it has exited.
func (s *Scheduler) send(ev event) bool {
	select {
	case <-s.stopped:
		return false
	default:
	}
	select {
	case s.events <- ev:
		return true
	case <-s.stopped:
		return false
	}
}

// drainAfterStop answers events that slipped into the channel while the
// loop was exiting so no caller blocks forever.
func (s *Scheduler) drainAfterStop() {
	for {
		select {
		case ev := <-s.events:
			switch e := ev.(type) {
			case evSubmit:
				e.js.handle.finish(ErrShuttingDown)
			case evStateReq:
				e.reply <- State{Now: s.opts.Now(), Mode: "stopped"}
			case evAdmin:
				e.reply <- ErrShuttingDown
			case evLoaded:
				if e.inst != nil {
					go func() { _ = e.inst.Stop(context.Background()) }()
				}
			}
		default:
			return
		}
	}
}

func (s *Scheduler) now() time.Time { return s.opts.Now() }

// Run executes the loop until ctx is cancelled, then drains and stops every
// instance. It returns nil on a clean shutdown.
func (s *Scheduler) Run(ctx context.Context) error {
	defer func() {
		close(s.stopped)
		s.drainAfterStop()
	}()

	// First snapshot synchronously: nothing can be admitted without a budget.
	// nvidia-smi can fail transiently while another process is tearing down
	// its GPU context, so try a few times before giving up on the fast path.
	var snapErr error
	for attempt := 1; attempt <= 3; attempt++ {
		if snapErr = s.snapshotSync(ctx); snapErr == nil {
			break
		}
		s.log.Debug("initial gpu snapshot failed", "attempt", attempt, "err", snapErr)
		select {
		case <-time.After(500 * time.Millisecond):
		case <-ctx.Done():
			s.shutdown()
			return nil
		}
	}
	if snapErr != nil {
		s.log.Warn("initial gpu snapshot failed; nothing will be admitted until polling succeeds", "err", snapErr)
	}
	s.preload()

	tick := time.NewTicker(s.opts.Tick)
	defer tick.Stop()
	poll := time.NewTicker(s.cfg.GPU.PollInterval)
	defer poll.Stop()

	for {
		select {
		case <-ctx.Done():
			s.shutdown()
			return nil
		case ev := <-s.events:
			s.handle(ev)
			s.schedule()
		case <-tick.C:
			s.checkDeadlines()
			s.schedule()
		case <-poll.C:
			s.pollGPU()
		}
	}
}

func (s *Scheduler) snapshotSync(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	snap, err := s.gpu.Snapshot(ctx)
	if err != nil {
		return err
	}
	s.onSnapshot(evSnapshot{snap: snap})
	return nil
}

func (s *Scheduler) pollGPU() {
	if s.polling {
		return
	}
	s.polling = true
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		snap, err := s.gpu.Snapshot(ctx)
		s.send(evSnapshot{snap: snap, err: err})
	}()
}

// preload loads pinned models first, then preload-only ones.
func (s *Scheduler) preload() {
	now := s.now()
	for _, pinnedPass := range []bool{true, false} {
		for _, id := range sortedSpecIDs(s.specs) {
			sp := s.specs[id]
			if sp.Pinned != pinnedPass || !sp.Preload {
				continue
			}
			status, err := s.ensureLoaded(sp, job.Background, now)
			if err != nil {
				if sp.Pinned {
					s.log.Error("pinned model cannot be loaded; it will never be available", "model", id, "err", err)
				} else {
					s.log.Warn("preload failed", "model", id, "err", err)
				}
				continue
			}
			s.log.Info("preload", "model", id, "status", status)
		}
	}
}

// shutdown fails queued jobs, waits for in-flight steps, stops instances.
func (s *Scheduler) shutdown() {
	s.shuttingDown = true
	s.event(EvShutdown, "scheduler", "draining")
	for _, js := range s.q.all() {
		s.fail(js, ErrShuttingDown, "shutdown")
	}
	deadline := time.NewTimer(s.opts.ShutdownTimeout)
	defer deadline.Stop()

	inflight := func() int {
		n := 0
		for _, js := range s.jobs {
			n += js.inflight
		}
		return n
	}
	for inflight() > 0 {
		select {
		case ev := <-s.events:
			s.handle(ev)
		case <-deadline.C:
			s.log.Warn("shutdown: giving up on in-flight steps", "inflight", inflight())
			goto stop
		}
	}
stop:
	pending := 0
	for _, e := range s.res.All() {
		if e.Instance != nil && e.State != residency.Stopping {
			// Past the drain deadline nothing is coming back; stop anyway
			// rather than leak the process to the next start's reaper.
			e.Running = 0
			s.stopEntry(e, "shutdown")
			pending++
		} else if e.State == residency.Stopping {
			pending++
		}
	}
	for pending > 0 {
		select {
		case ev := <-s.events:
			if st, ok := ev.(evStopped); ok {
				s.handle(st)
				pending--
			} else if _, ok := ev.(evLoaded); ok {
				// a load finishing during shutdown: stop it too
				s.handle(ev)
				if e, ok := s.res.Get(ev.(evLoaded).id); ok && e.Instance != nil {
					s.stopEntry(e, "shutdown")
					pending++
				}
			} else {
				s.handle(ev)
			}
		case <-deadline.C:
			s.log.Warn("shutdown: instances still stopping", "pending", pending)
			return
		}
	}
}
