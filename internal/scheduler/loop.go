package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/w512/gridcore/internal/gpu"
	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/residency"
	"github.com/w512/gridcore/internal/runtime"
)

// event is anything the loop reacts to. All mutation happens in handle().
type event interface{ isEvent() }

type evSubmit struct{ js *jobState }
type evStepDone struct {
	id    string
	step  int
	err   error
	usage Usage
}
type evCancel struct{ id string }
type evLoaded struct {
	id   string
	inst runtime.Instance
	err  error
}
type evDied struct {
	id   string
	inst runtime.Instance
}
type evStopped struct {
	id     string
	inst   runtime.Instance
	reason string
}
type evSnapshot struct {
	snap gpu.Snapshot
	err  error
}
type evStateReq struct{ reply chan State }
type evAdmin struct {
	op    string
	model string
	reply chan error
}

func (evSubmit) isEvent()   {}
func (evStepDone) isEvent() {}
func (evCancel) isEvent()   {}
func (evLoaded) isEvent()   {}
func (evDied) isEvent()     {}
func (evStopped) isEvent()  {}
func (evSnapshot) isEvent() {}
func (evStateReq) isEvent() {}
func (evAdmin) isEvent()    {}

func (s *Scheduler) handle(ev event) {
	switch e := ev.(type) {
	case evSubmit:
		s.onSubmit(e)
	case evStepDone:
		s.onStepDone(e)
	case evCancel:
		s.onCancel(e)
	case evLoaded:
		s.onLoaded(e)
	case evDied:
		s.onDied(e)
	case evStopped:
		s.onStopped(e)
	case evSnapshot:
		s.onSnapshot(e)
	case evStateReq:
		e.reply <- s.buildState()
	case evAdmin:
		e.reply <- s.onAdmin(e)
	}
}

// ---- jobs ----

func (s *Scheduler) onSubmit(e evSubmit) {
	js := e.js
	if s.shuttingDown {
		js.handle.finish(ErrShuttingDown)
		return
	}
	s.jobs[js.job.ID] = js
	js.job.State = job.Queued
	js.lastProgress = js.job.Enqueued // starvation is measured from arrival
	s.q.push(js)
	s.event(EvEnqueue, js.job.ID, fmt.Sprintf("%s %s %s steps=%d", js.job.Class, js.job.Kind, js.job.Target(), js.job.Steps))
}

func (s *Scheduler) onStepDone(e evStepDone) {
	js, ok := s.jobs[e.id]
	if !ok {
		return // job already gone (e.g. crash cleanup); nothing to account
	}
	now := s.now()
	entryID, had := js.steps[e.step]
	if !had {
		return // duplicate or unknown step
	}
	delete(js.steps, e.step)
	js.inflight--
	js.lastProgress = now
	if s.guard.active && s.guard.job == e.id && s.guard.step == e.step {
		// A step of length d at share f is followed by d*(1/f - 1) without
		// one, so lower-class work takes f of the time whatever its step
		// length. f = 1 lets the next step in right away.
		s.guard.active = false
		if f := s.cfg.Policy.BackgroundShareOrDefault(); f < 1 {
			d := now.Sub(s.guard.start)
			s.nextLowerAt = now.Add(time.Duration(float64(d) * (1/f - 1)))
		}
	}
	if ent, ok := s.res.Get(entryID); ok && ent.State != residency.Loading {
		if ent.Running > 0 {
			ent.Running--
		}
		if ent.State == residency.Draining && ent.Running == 0 {
			s.stopEntry(ent, "evicted")
		}
	}
	if js.job.Class == job.Interactive {
		if s.runningInteractive > 0 {
			s.runningInteractive--
		}
		s.lastInteractiveEnd = now
	}
	if start, ok := js.stepStart[e.step]; ok {
		delete(js.stepStart, e.step)
		s.m.ExecutionSeconds.WithLabelValues(string(js.job.Class), js.job.ModelID).Observe(now.Sub(start).Seconds())
	}
	s.recordUsage(js.job.ModelID, e.usage)

	if e.err != nil {
		if !js.failed {
			outcome := "step_error"
			var sc interface{ StatusCode() int }
			if errors.As(e.err, &sc) && sc.StatusCode() >= 400 && sc.StatusCode() < 500 {
				outcome = "client_error" // the runtime rejected the request (e.g. context too long)
			}
			s.fail(js, fmt.Errorf("%w: step %d: %v", ErrStepFailed, e.step, e.err), outcome)
		}
	} else if !js.failed {
		js.completed++
		if js.completed == js.job.Steps {
			s.complete(js)
		}
	}
	if js.failed && js.inflight == 0 {
		delete(s.jobs, js.job.ID)
	}
}

func (s *Scheduler) onCancel(e evCancel) {
	js, ok := s.jobs[e.id]
	if !ok || js.failed {
		return
	}
	phase := "running"
	if js.dispatched == 0 {
		phase = "queued"
	}
	s.m.JobsCancelled.WithLabelValues(string(js.job.Class), phase).Inc()
	s.fail(js, ErrCancelled, "cancelled")
}

// fail marks a job terminal with err. Slots held by in-flight steps are
// returned as their StepDone events arrive.
func (s *Scheduler) fail(js *jobState, err error, outcome string) {
	if js.failed {
		return
	}
	js.failed = true
	js.failErr = err
	s.q.remove(js)
	switch {
	case errors.Is(err, ErrQueueTimeout):
		js.job.State = job.TimedOut
		s.event(EvTimeout, js.job.ID, err.Error())
	case errors.Is(err, ErrCancelled):
		js.job.State = job.Cancelled
		s.event(EvCancel, js.job.ID, "")
	default:
		js.job.State = job.Failed
		s.event(EvFail, js.job.ID, err.Error())
	}
	s.m.JobsTotal.WithLabelValues(string(js.job.Class), string(js.job.Kind), outcome).Inc()
	js.handle.finish(err)
	if js.inflight == 0 {
		delete(s.jobs, js.job.ID)
	}
}

func (s *Scheduler) complete(js *jobState) {
	js.job.State = job.Done
	s.m.JobsTotal.WithLabelValues(string(js.job.Class), string(js.job.Kind), "ok").Inc()
	s.event(EvComplete, js.job.ID, fmt.Sprintf("%s %s", js.job.Class, js.job.ModelID))
	js.handle.finish(nil)
	delete(s.jobs, js.job.ID)
}

// failQueuedFor fails every not-yet-dispatched job that targets model id.
func (s *Scheduler) failQueuedFor(id string, err error, outcome string) {
	for _, js := range s.q.all() {
		if js.job.ModelID == id && js.dispatched == 0 {
			s.fail(js, err, outcome)
		}
	}
}

func (s *Scheduler) checkDeadlines() {
	now := s.now()
	for _, js := range s.q.all() {
		if js.dispatched > 0 {
			continue
		}
		if dl, ok := js.job.Deadline(); ok && !now.Before(dl) {
			s.fail(js, ErrQueueTimeout, "timeout")
		}
	}
}

func (s *Scheduler) recordUsage(modelID string, u Usage) {
	if u.PromptTokens > 0 {
		s.m.TokensTotal.WithLabelValues(modelID, "prompt").Add(float64(u.PromptTokens))
	}
	if u.CompletionTokens > 0 {
		s.m.TokensTotal.WithLabelValues(modelID, "generated").Add(float64(u.CompletionTokens))
	}
	if u.PromptTPS > 0 || u.GenTPS > 0 {
		if sp, ok := s.specs[modelID]; ok {
			s.updateProfile(sp, func(p *model.Profile) { p.ObserveThroughput(u.PromptTPS, u.GenTPS) })
		}
	}
}

// ---- instances ----

func (s *Scheduler) onLoaded(e evLoaded) {
	ent, ok := s.res.Get(e.id)
	if !ok {
		// Entry vanished (shutdown raced the load). Do not leak the process.
		if e.inst != nil {
			go func() { _ = e.inst.Stop(context.Background()) }()
		}
		return
	}
	now := s.now()
	dur := now.Sub(ent.LoadStarted)
	if e.err != nil {
		s.res.Remove(e.id)
		s.ports[ent.Spec.Runtime].release(ent.Port)
		outcome := "error"
		if errors.Is(e.err, runtime.ErrLoadTimeout) {
			outcome = "timeout"
		}
		s.event(EvLoadFail, e.id, e.err.Error())
		attrs := []any{"model", e.id, "err", e.err, "took", dur, "reserved_mb", ent.VRAMMB}
		var det runtime.Detailed
		if errors.As(e.err, &det) {
			attrs = append(attrs, "output", det.Tail())
		}
		s.log.Error("model load failed", attrs...)
		// Out of memory means the admission size was wrong: forget the stored
		// measurement so the next attempt starts from a fresh estimate, and
		// remember to be more careful.
		var oom runtime.OOMError
		if errors.As(e.err, &oom) && oom.OOM() {
			outcome = "oom"
			if p, ok := s.store.Get(s.profileKey(ent.Spec)); ok && p.VRAMMB > 0 {
				s.updateProfile(ent.Spec, func(p *model.Profile) { p.VRAMMB = 0 })
				s.event(EvLoadFail, e.id, fmt.Sprintf("out of memory with reserved %d MB; stored profile (%d MB) discarded", ent.VRAMMB, p.VRAMMB))
				s.log.Warn("profile discarded after OOM", "model", e.id, "profile_mb", p.VRAMMB, "reserved_mb", ent.VRAMMB)
			}
		}
		s.m.ModelLoadsTotal.WithLabelValues(e.id, outcome).Inc()
		// Jobs that triggered this load learn the real cause; only later
		// arrivals see the breaker.
		s.failQueuedFor(e.id, fmt.Errorf("%w: %v", ErrLoadFailed, e.err), "load_error")
		s.recordFailure(e.id, "load failed: "+e.err.Error())
		s.updateResidentGauge()
		return
	}
	ent.Instance = e.inst
	ent.State = residency.Ready
	ent.LoadedAt = now
	if ent.LastUsed.IsZero() {
		ent.LastUsed = now // never-used models are LRU-ordered by load time
	}
	s.updateProfile(ent.Spec, func(p *model.Profile) { p.ObserveLoad(dur) })
	if p, ok := s.store.Get(s.profileKey(ent.Spec)); ok && p.VRAMMB > 0 {
		ent.VRAMMB = p.VRAMMB
	}
	s.m.ModelLoadSeconds.WithLabelValues(e.id).Observe(dur.Seconds())
	s.m.ModelLoadsTotal.WithLabelValues(e.id, "ok").Inc()
	s.event(EvLoaded, e.id, fmt.Sprintf("%.1fs pid=%d addr=%s", dur.Seconds(), e.inst.PID(), e.inst.Addr()))
	s.log.Info("model loaded", "model", e.id, "took", dur, "addr", e.inst.Addr(), "vram_mb", ent.VRAMMB)
	s.updateResidentGauge()
	s.checkThrash(e.id, now)

	inst := e.inst
	go func() {
		<-inst.Done()
		s.send(evDied{id: e.id, inst: inst})
	}()
}

func (s *Scheduler) onDied(e evDied) {
	ent, ok := s.res.Get(e.id)
	if !ok || ent.Instance != e.inst || ent.State == residency.Stopping {
		return // expected exit (we stopped it) or already replaced
	}
	s.res.Remove(e.id)
	s.ports[ent.Spec.Runtime].release(ent.Port)
	s.externalSettle = externalSettleSnapshots
	s.m.InstanceCrashes.WithLabelValues(e.id).Inc()
	detail := "process exited"
	if err := e.inst.Err(); err != nil {
		detail = err.Error()
	}
	s.event(EvCrash, e.id, detail)
	s.log.Error("model instance died", "model", e.id, "detail", detail)
	s.recordFailure(e.id, "crash: "+detail)
	// Steps in flight on this instance will fail on their own; make the
	// outcome explicit so the API layer can stop waiting for other steps.
	for _, js := range s.jobs {
		for _, entryID := range js.steps {
			if entryID == e.id && !js.failed {
				s.fail(js, fmt.Errorf("%w: %s", ErrInstanceFailed, detail), "instance_failed")
				break
			}
		}
	}
	s.updateResidentGauge()
}

func (s *Scheduler) onStopped(e evStopped) {
	ent, ok := s.res.Get(e.id)
	if !ok || ent.Instance != e.inst {
		return
	}
	s.res.Remove(e.id)
	s.ports[ent.Spec.Runtime].release(ent.Port)
	s.externalSettle = externalSettleSnapshots
	s.m.ModelEvictions.WithLabelValues(e.id, e.reason).Inc()
	s.event(EvUnloaded, e.id, e.reason)
	s.log.Info("model unloaded", "model", e.id, "reason", e.reason)
	s.updateResidentGauge()
}

// externalSettleSnapshots is how many polls after an instance disappears
// its memory may still show up as unattributed. One covers a snapshot that
// was already in flight; the second covers driver teardown lag.
const externalSettleSnapshots = 2

// stopEntry transitions a drained entry to Stopping and stops it
// asynchronously. Callers guarantee Running == 0 (invariant #1).
func (s *Scheduler) stopEntry(ent *residency.Entry, reason string) {
	if ent.State == residency.Stopping || ent.Instance == nil {
		return
	}
	if ent.Running > 0 {
		s.log.Error("BUG: stopEntry with running steps", "model", ent.ID, "running", ent.Running)
		return
	}
	ent.State = residency.Stopping
	inst := ent.Instance
	id := ent.ID
	go func() {
		if err := inst.Stop(context.Background()); err != nil {
			s.log.Warn("instance stop", "model", id, "err", err)
		}
		s.send(evStopped{id: id, inst: inst, reason: reason})
	}()
}

// ---- GPU ----

func (s *Scheduler) onSnapshot(e evSnapshot) {
	s.polling = false
	if e.err != nil {
		// Log the first failure and then every 20th (10s at the default
		// poll interval) so a broken nvidia-smi does not flood the journal.
		s.snapFailures++
		if s.snapFailures == 1 || s.snapFailures%20 == 0 {
			s.log.Warn("gpu snapshot failed", "err", e.err, "consecutive", s.snapFailures)
		}
		return
	}
	if s.snapFailures > 0 {
		s.log.Info("gpu snapshot recovered", "after_failures", s.snapFailures)
		s.snapFailures = 0
	}
	s.snap = e.snap
	s.snapOK = true
	if s.gpuName == "" {
		s.gpuName = e.snap.Name
		s.migrateProfiles()
	}

	ours := 0
	transient := false // an instance is appearing or disappearing
	for _, ent := range s.res.All() {
		if ent.State == residency.Loading {
			transient = true
			continue
		}
		if ent.State == residency.Stopping {
			transient = true
		}
		if ent.Instance == nil {
			continue
		}
		mb := e.snap.UsedByPID(ent.Instance.PID())
		ours += mb
		if mb <= 0 {
			continue
		}
		if !ent.Measured || mb > ent.VRAMMB {
			ent.VRAMMB = mb
		}
		ent.Measured = true
		s.updateProfile(ent.Spec, func(p *model.Profile) { p.ObserveVRAM(mb) })
	}
	// Memory we cannot attribute to our own instances is "external". While
	// an instance is loading its growing footprint would land there; while
	// one is stopping (and for a couple of polls after it is gone) the
	// driver may still report its memory as used after the process has left
	// the per-PID list. In both cases hold the last known value rather than
	// mistake our own memory for someone else's and evict a neighbour to
	// make room for it. A drop is always real and is taken immediately.
	if s.externalSettle > 0 {
		s.externalSettle--
		transient = true
	}
	ext := e.snap.UsedMB - ours
	if ext < 0 {
		ext = 0
	}
	if !transient || ext < s.externalMB {
		s.externalMB = ext
	}

	mb := func(v int) float64 { return float64(v) * 1024 * 1024 }
	s.m.GPUVRAMUsedBytes.Set(mb(e.snap.UsedMB))
	s.m.GPUVRAMTotalBytes.Set(mb(s.budgetMB()))
	s.m.GPUVRAMReserved.Set(mb(s.reservedMB()))
	if e.snap.UtilPct >= 0 {
		s.m.GPUUtilization.Set(float64(e.snap.UtilPct))
	}
}

// ---- admin ----

func (s *Scheduler) onAdmin(e evAdmin) error {
	sp := s.specs[e.model]
	now := s.now()
	switch e.op {
	case "load":
		status, err := s.ensureLoaded(sp, job.Background, now)
		if err != nil {
			return err
		}
		if status == "waiting" {
			return errors.New("insufficient VRAM and nothing evictable")
		}
		return nil
	case "unload":
		ent, ok := s.res.Get(e.model)
		if !ok {
			return errors.New("model is not resident")
		}
		if sp.Pinned {
			return errors.New("model is pinned")
		}
		if ent.State != residency.Ready {
			return fmt.Errorf("model is %s", ent.State)
		}
		s.evict(ent, "admin")
		return nil
	case "enable":
		delete(s.disabled, e.model)
		delete(s.breakers, e.model)
		s.event(EvEnabled, e.model, "")
		return nil
	}
	return fmt.Errorf("unknown admin op %q", e.op)
}

// ---- thrash ----

// A model loaded thrashLoads times within thrashWindow is being pushed out
// and brought back over and over: the models in use do not fit together.
// The scheduler cannot fix that (min_residency only slows it down), so it
// says so, once per window.
const (
	thrashWindow = 5 * time.Minute
	thrashLoads  = 6
)

func (s *Scheduler) checkThrash(id string, now time.Time) {
	kept := s.loads[id][:0]
	for _, t := range s.loads[id] {
		if now.Sub(t) < thrashWindow {
			kept = append(kept, t)
		}
	}
	s.loads[id] = append(kept, now)
	if len(s.loads[id]) < thrashLoads {
		return
	}
	if last, ok := s.thrashAt[id]; ok && now.Sub(last) < thrashWindow {
		return
	}
	s.thrashAt[id] = now
	s.m.ModelThrash.WithLabelValues(id).Inc()
	detail := fmt.Sprintf("loaded %d times in %s: the models in use do not fit together; lower ctx or parallel of some of them, or headroom_mb", len(s.loads[id]), thrashWindow)
	s.event(EvThrash, id, detail)
	s.log.Warn("model thrash", "model", id, "loads", len(s.loads[id]), "window", thrashWindow)
}

// ---- breaker ----

type breaker struct{ failures []time.Time }

func (s *Scheduler) recordFailure(id, detail string) {
	now := s.now()
	b := s.breakers[id]
	if b == nil {
		b = &breaker{}
		s.breakers[id] = b
	}
	kept := b.failures[:0]
	for _, t := range b.failures {
		if now.Sub(t) < s.opts.BreakerWindow {
			kept = append(kept, t)
		}
	}
	b.failures = append(kept, now)
	if len(b.failures) >= s.opts.BreakerThreshold {
		if _, already := s.disabled[id]; !already {
			s.disabled[id] = detail
			s.event(EvDisabled, id, fmt.Sprintf("%d failures within %s; last: %s", len(b.failures), s.opts.BreakerWindow, detail))
			s.log.Error("model disabled by circuit breaker", "model", id, "failures", len(b.failures))
		}
		s.failQueuedFor(id, ErrModelDisabled, "disabled")
	}
}

// ---- helpers ----

func (s *Scheduler) event(kind, subject, detail string) {
	s.ring.add(Event{At: s.now(), Kind: kind, Subject: subject, Detail: detail})
}

func (s *Scheduler) profileKey(sp *model.Spec) string {
	return sp.ProfileKey(s.binaryIDs[sp.Runtime], s.gpuName)
}

// migrateProfiles carries measurements over from the 0.1 key, which named
// the runtime by its binary's path, mtime and size. A profile found under
// that key was recorded with exactly the binary on disk now, so it is as
// good as one recorded under the build id.
func (s *Scheduler) migrateProfiles() {
	for _, id := range sortedSpecIDs(s.specs) {
		sp := s.specs[id]
		legacy := binaryID(s.cfg.Runtimes[sp.Runtime])
		if legacy == s.binaryIDs[sp.Runtime] {
			continue // the runtime reports no version; keys are unchanged
		}
		old, ok := s.store.Get(sp.ProfileKey(legacy, s.gpuName))
		if !ok {
			continue
		}
		if _, have := s.store.Get(s.profileKey(sp)); have {
			continue
		}
		s.updateProfile(sp, func(p *model.Profile) {
			p.VRAMMB, p.LoadMS, p.GenTPS, p.PromptTPS = old.VRAMMB, old.LoadMS, old.GenTPS, old.PromptTPS
			p.Samples, p.UpdatedAt = old.Samples, old.UpdatedAt
		})
		s.log.Info("profile carried over to the runtime build key", "model", id, "runtime", s.binaryIDs[sp.Runtime], "vram_mb", old.VRAMMB)
	}
}

// updateProfile applies fn to sp's profile, recording which runtime build
// and GPU it belongs to.
func (s *Scheduler) updateProfile(sp *model.Spec, fn func(*model.Profile)) {
	rt, gpuName := s.binaryIDs[sp.Runtime], s.gpuName
	err := s.store.Update(sp.ProfileKey(rt, gpuName), sp.ID, func(p *model.Profile) {
		p.Runtime, p.GPU = rt, gpuName
		fn(p)
	})
	// Updates arrive with every GPU snapshot: report a broken file once,
	// not twice a second.
	if err != nil && !s.profileErr {
		s.log.Warn("profile update failed; measurements are kept in memory only", "model", sp.ID, "err", err)
	}
	s.profileErr = err != nil
}

func (s *Scheduler) updateResidentGauge() {
	n := 0
	for _, e := range s.res.All() {
		if e.State == residency.Ready || e.State == residency.Draining {
			n++
		}
	}
	s.m.ModelsResident.Set(float64(n))
	s.m.GPUVRAMReserved.Set(float64(s.reservedMB()) * 1024 * 1024)
}

func sortedSpecIDs(specs map[string]*model.Spec) []string {
	ids := make([]string, 0, len(specs))
	for id := range specs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}
