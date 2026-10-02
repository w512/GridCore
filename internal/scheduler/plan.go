package scheduler

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/w512/gridcore/internal/config"
	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/residency"
)

// schedule runs one planning pass over the queues. It is called after every
// event and on every tick, so it must be cheap and idempotent.
func (s *Scheduler) schedule() {
	if s.shuttingDown {
		return
	}
	now := s.now()

	// Interactive jobs are always considered, each independently: one waiting
	// for a load must not block another whose model is resident.
	for _, js := range snapshotList(s.q.list(job.Interactive)) {
		s.planJob(js, now)
	}

	interactive := s.interactiveMode(now)
	s.setMode(interactive, now)
	if interactive {
		// Bounded starvation: once lower-class jobs have waited too long, let
		// a single step through even though interactive work is active. Its
		// impact is one small step's worth of contention. Jobs are tried
		// oldest-progress first, but one whose model cannot be made resident
		// (e.g. it would need to evict a hot model) must not block others
		// whose model is already loaded. background_share spaces these steps
		// out (see onStepDone).
		if s.runningLowerSteps() == 0 && !now.Before(s.nextLowerAt) {
			for _, js := range s.starvedJobs(now) {
				before := js.dispatched
				s.planJobLimited(js, now, 1)
				if js.dispatched > before {
					s.guard = guardStep{job: js.job.ID, step: before, start: now, active: true}
					s.event(EvPreempt, js.job.ID, fmt.Sprintf("%s starved %s; running one step alongside interactive", js.job.Class, now.Sub(js.lastProgress).Round(time.Millisecond)))
					break
				}
			}
		}
		reason := "interactive mode"
		if now.Before(s.nextLowerAt) {
			reason = "interactive mode; background share, next step in " + s.nextLowerAt.Sub(now).Round(100*time.Millisecond).String()
		}
		for _, js := range s.q.list(job.Background) {
			if js.reason == "" || js.remaining() > 0 {
				js.reason = reason
			}
		}
		for _, js := range s.q.list(job.Batch) {
			js.reason = reason
		}
	} else {
		for _, js := range snapshotList(s.q.list(job.Background)) {
			s.planJob(js, now)
		}
		// Batch only runs when background has nothing waiting.
		if s.q.len(job.Background) == 0 {
			for _, js := range snapshotList(s.q.list(job.Batch)) {
				s.planJob(js, now)
			}
		} else {
			for _, js := range s.q.list(job.Batch) {
				js.reason = "background queued"
			}
		}
	}
	s.updateGauges()
}

// snapshotList copies a queue slice so planJob may remove entries while we
// iterate.
func snapshotList(l []*jobState) []*jobState {
	out := make([]*jobState, len(l))
	copy(out, l)
	return out
}

// interactiveMode reports whether background/batch steps must not start.
// Queued interactive jobs count: they are about to run and should not have
// to compete with freshly started background steps.
func (s *Scheduler) interactiveMode(now time.Time) bool {
	if s.runningInteractive > 0 || s.q.len(job.Interactive) > 0 {
		return true
	}
	idle := s.cfg.Policy.InteractiveIdleBeforeBackground
	return !s.lastInteractiveEnd.IsZero() && now.Sub(s.lastInteractiveEnd) < idle
}

func (s *Scheduler) setMode(interactive bool, now time.Time) {
	mode := "idle"
	if interactive {
		mode = "interactive"
	}
	if mode != s.mode {
		s.mode = mode
		s.event(EvMode, "gpu", mode)
	}
}

// starvedJobs returns background/batch jobs that have made no progress for
// longer than policy.background_max_starvation, oldest first.
func (s *Scheduler) starvedJobs(now time.Time) []*jobState {
	limit := s.cfg.Policy.MaxStarvation()
	if limit <= 0 {
		return nil
	}
	var out []*jobState
	for _, c := range []job.Class{job.Background, job.Batch} {
		for _, js := range s.q.list(c) {
			if js.failed || js.remaining() == 0 || now.Sub(js.lastProgress) < limit {
				continue
			}
			out = append(out, js)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].lastProgress.Before(out[j].lastProgress) })
	return out
}

// runningLowerSteps counts in-flight background/batch steps.
func (s *Scheduler) runningLowerSteps() int {
	n := 0
	for _, js := range s.jobs {
		if js.job.Class != job.Interactive {
			n += js.inflight
		}
	}
	return n
}

// planJob tries to dispatch as many steps of js as possible right now, or
// starts whatever is needed (load, eviction) so it can run later.
func (s *Scheduler) planJob(js *jobState, now time.Time) {
	s.planJobLimited(js, now, 0)
}

// planJobLimited is planJob with a cap on steps dispatched in this pass
// (0 = no cap).
func (s *Scheduler) planJobLimited(js *jobState, now time.Time, maxSteps int) {
	if js.failed || js.remaining() == 0 {
		return
	}
	id := js.job.ModelID
	if reason, off := s.disabled[id]; off {
		s.fail(js, fmt.Errorf("%w: %s", ErrModelDisabled, reason), "disabled")
		return
	}
	sp := s.specs[id]

	if ent, ok := s.res.Get(id); ok {
		switch ent.State {
		case residency.Loading:
			js.reason = "model loading"
		case residency.Draining, residency.Stopping:
			js.reason = "model unloading"
		case residency.Ready:
			n := 0
			for js.remaining() > 0 && ent.FreeSlots() > 0 && (maxSteps == 0 || n < maxSteps) {
				s.dispatch(js, ent, now)
				n++
			}
			if js.remaining() > 0 {
				js.reason = "waiting for slot"
			}
		}
		return
	}

	status, err := s.ensureLoaded(sp, js.job.Class, now)
	if err != nil {
		s.fail(js, err, "admission")
		return
	}
	switch status {
	case "loading":
		js.reason = "model loading"
	case "evicting":
		js.reason = "evicting to make room"
	default:
		js.reason = "waiting for VRAM" + s.heldBy(js.job.Class, now)
	}
}

// heldBy names the resident models that the residency rules keep from
// being evicted for class c, e.g. " (kept: e4b in use by background)".
// Pinned models are left out: they are always kept.
func (s *Scheduler) heldBy(c job.Class, now time.Time) string {
	var held []string
	for _, e := range s.res.All() {
		if e.State != residency.Ready || e.Evicting {
			continue
		}
		if p, why := s.res.Protected(e, c, now); p && why != "pinned" {
			held = append(held, e.ID+" "+why)
		}
	}
	if len(held) == 0 {
		return ""
	}
	return " (kept: " + strings.Join(held, ", ") + ")"
}

// ensureLoaded admits a model that is not resident. It returns one of:
//
//	"ready"    already resident
//	"loading"  a load was started (or is in progress)
//	"evicting" victims were marked; the load starts once they are gone
//	"waiting"  nothing can be done right now
//
// or an error when the model can never fit.
func (s *Scheduler) ensureLoaded(sp *model.Spec, c job.Class, now time.Time) (string, error) {
	if ent, ok := s.res.Get(sp.ID); ok {
		if ent.State == residency.Loading {
			return "loading", nil
		}
		return "ready", nil
	}
	if !s.snapOK {
		return "waiting", nil
	}
	need := s.needMB(sp)
	if need > s.maxLoadableMB(sp) {
		return "", fmt.Errorf("%w: needs %d MB, at most %d MB can ever be free", ErrModelTooLarge, need, s.maxLoadableMB(sp))
	}
	avail := s.availableMB()
	if need <= avail {
		if err := s.startLoad(sp, need, c, now); err != nil {
			return "", err
		}
		return "loading", nil
	}
	// Memory of instances already being evicted is on its way back; do not
	// pick further victims while it would be enough.
	if pending := s.pendingFreeMB(); need <= avail+pending {
		return "evicting", nil
	} else {
		avail += pending
	}
	victims := s.res.Victims(need-avail, c, now)
	if victims == nil {
		return "waiting", nil
	}
	for _, v := range victims {
		reason := "make room for " + sp.ID
		if s.cfg.Policy.Eviction == config.EvictionCost {
			reason += fmt.Sprintf(", cost %.2g", s.res.Cost(v, now))
		}
		s.evict(v, reason)
	}
	return "evicting", nil
}

// reloadSeconds is the stored load time of e's model, or 0 when unknown.
func (s *Scheduler) reloadSeconds(e *residency.Entry) float64 {
	if p, ok := s.store.Get(s.profileKey(e.Spec)); ok {
		return p.LoadMS / 1000
	}
	return 0
}

// pendingFreeMB is the footprint of entries marked for eviction that have
// not been removed yet (draining or stopping).
func (s *Scheduler) pendingFreeMB() int {
	n := 0
	for _, e := range s.res.All() {
		if e.Evicting {
			n += e.VRAMMB
		}
	}
	return n
}

func (s *Scheduler) startLoad(sp *model.Spec, needMB int, c job.Class, now time.Time) error {
	rt, ok := s.runtimes[sp.Runtime]
	if !ok {
		return fmt.Errorf("no runtime %q", sp.Runtime)
	}
	port, err := s.ports[sp.Runtime].alloc()
	if err != nil {
		return err
	}
	ent := &residency.Entry{
		ID:          sp.ID,
		Spec:        sp,
		State:       residency.Loading,
		Port:        port,
		VRAMMB:      needMB,
		Slots:       sp.Parallel,
		LoadedFor:   c,
		LoadStarted: now,
	}
	s.res.Add(ent)
	s.event(EvLoad, sp.ID, fmt.Sprintf("reserve %d MB port=%d", needMB, port))
	s.log.Info("loading model", "model", sp.ID, "reserve_mb", needMB, "port", port)
	s.m.GPUVRAMReserved.Set(float64(s.reservedMB()) * 1024 * 1024)

	timeout := s.cfg.Runtimes[sp.Runtime].LoadTimeout
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		inst, err := rt.Load(ctx, sp, port)
		s.send(evLoaded{id: sp.ID, inst: inst, err: err})
	}()
	return nil
}

// evict marks ent as a victim. It is stopped now if idle, otherwise when its
// last running step completes.
func (s *Scheduler) evict(ent *residency.Entry, reason string) {
	if ent.Evicting {
		return
	}
	ent.Evicting = true
	ent.State = residency.Draining
	s.event(EvEvict, ent.ID, fmt.Sprintf("%s (running=%d)", reason, ent.Running))
	s.log.Info("evicting model", "model", ent.ID, "reason", reason, "running", ent.Running)
	if ent.Running == 0 {
		s.stopEntry(ent, "evicted")
	}
}

func (s *Scheduler) dispatch(js *jobState, ent *residency.Entry, now time.Time) {
	step := js.dispatched
	js.dispatched++
	js.inflight++
	js.steps[step] = ent.ID
	js.stepStart[step] = now
	ent.Running++
	s.res.Touch(ent.ID, js.job.Class, now)
	if step == 0 {
		s.res.Use(ent.ID, js.job.Class, now) // demand counts requests, not chunks
	}
	if js.job.Class == job.Interactive {
		s.runningInteractive++
	}
	if js.firstGrant.IsZero() {
		js.firstGrant = now
		js.job.State = job.Running
		s.m.QueueSeconds.WithLabelValues(string(js.job.Class)).Observe(now.Sub(js.job.Enqueued).Seconds())
	}
	if js.remaining() == 0 {
		s.q.remove(js)
	}
	js.reason = ""
	js.handle.grants <- Grant{
		Step:     step,
		Model:    ent.ID,
		Addr:     ent.Instance.Addr(),
		Instance: fmt.Sprintf("%s:%d", ent.ID, ent.Port),
		Queued:   js.firstGrant.Sub(js.job.Enqueued),
	}
	if step == 0 || js.job.Steps > 1 && step == js.job.Steps-1 {
		s.event(EvDispatch, js.job.ID, fmt.Sprintf("%s step %d/%d -> %s", js.job.Class, step+1, js.job.Steps, ent.ID))
	}
}

// ---- VRAM accounting ----

// budgetMB is what the scheduler may hand out in total.
func (s *Scheduler) budgetMB() int {
	if !s.snapOK {
		return 0
	}
	total := s.snap.TotalMB
	if s.cfg.GPU.VRAMLimitMB != nil && *s.cfg.GPU.VRAMLimitMB < total {
		total = *s.cfg.GPU.VRAMLimitMB
	}
	if b := total - s.cfg.GPU.HeadroomMB; b > 0 {
		return b
	}
	return 0
}

// committedMB is everything already spoken for: reservations, resident
// footprints and memory used by processes we do not manage.
func (s *Scheduler) committedMB() int { return s.res.CommittedMB() + s.externalMB }

func (s *Scheduler) availableMB() int {
	if a := s.budgetMB() - s.committedMB(); a > 0 {
		return a
	}
	return 0
}

func (s *Scheduler) reservedMB() int {
	n := 0
	for _, e := range s.res.All() {
		if e.State == residency.Loading {
			n += e.VRAMMB
		}
	}
	return n
}

// maxLoadableMB is the most sp could ever get: the budget minus every other
// pinned model (resident or not) and external usage.
func (s *Scheduler) maxLoadableMB(sp *model.Spec) int {
	mx := s.budgetMB() - s.externalMB
	for id, other := range s.specs {
		if id == sp.ID || !other.Pinned {
			continue
		}
		mx -= s.needMB(other)
	}
	if mx < 0 {
		return 0
	}
	return mx
}

// needMB is the admission size for sp: the stored measurement when we have
// one, otherwise the estimate padded with headroom.
func (s *Scheduler) needMB(sp *model.Spec) int {
	if p, ok := s.store.Get(s.profileKey(sp)); ok && p.VRAMMB > 0 {
		return p.VRAMMB
	}
	est := sp.EstimateVRAMMB()
	if est <= 0 {
		s.log.Warn("no VRAM estimate for model; using headroom as placeholder", "model", sp.ID)
		est = s.cfg.GPU.HeadroomMB
	}
	return est + s.cfg.GPU.HeadroomMB
}

func (s *Scheduler) updateGauges() {
	for _, c := range job.Classes {
		s.m.JobsQueued.WithLabelValues(string(c)).Set(float64(s.q.len(c)))
	}
	running := map[job.Class]int{}
	for _, js := range s.jobs {
		if js.inflight > 0 {
			running[js.job.Class]++
		}
	}
	for _, c := range job.Classes {
		s.m.JobsRunning.WithLabelValues(string(c)).Set(float64(running[c]))
	}
}

// ---- state snapshot ----

func (s *Scheduler) buildState() State {
	now := s.now()
	st := State{Now: now, Mode: s.mode, Events: s.ring.list()}
	st.GPU = GPUState{
		Name:        s.snap.Name,
		TotalMB:     s.snap.TotalMB,
		BudgetMB:    s.budgetMB(),
		UsedMB:      s.snap.UsedMB,
		CommittedMB: s.committedMB(),
		AvailableMB: s.availableMB(),
		ExternalMB:  s.externalMB,
		UtilPct:     s.snap.UtilPct,
		SnapshotAt:  s.snap.At,
	}
	for _, e := range s.res.All() {
		rm := ResidentModel{
			ID:              e.ID,
			State:           string(e.State),
			Tier:            string(s.res.Tier(e, now)),
			VRAMMB:          e.VRAMMB,
			Measured:        e.Measured,
			Slots:           e.Slots,
			BusySlots:       e.Running,
			LoadedAt:        e.LoadedAt,
			LastUsed:        e.LastUsed,
			LastInteractive: e.LastInteractive,
			EvictCost:       s.res.Cost(e, now),
		}
		if e.Instance != nil {
			rm.PID = e.Instance.PID()
			rm.Addr = e.Instance.Addr()
			rm.LoadSeconds = e.LoadedAt.Sub(e.LoadStarted).Seconds()
		}
		st.Resident = append(st.Resident, rm)
	}
	for _, js := range s.jobs {
		if js.inflight == 0 && js.dispatched == 0 {
			continue
		}
		st.Running = append(st.Running, s.jobState(js, now))
	}
	for _, js := range s.q.all() {
		if js.dispatched > 0 {
			continue // partially dispatched jobs show under Running
		}
		st.Queued = append(st.Queued, s.jobState(js, now))
	}
	for id := range s.disabled {
		st.Disabled = append(st.Disabled, id)
	}
	sortJobStates(st.Running)
	sortStrings(st.Disabled)
	return st
}

func (s *Scheduler) jobState(js *jobState, now time.Time) JobState {
	waited := now.Sub(js.job.Enqueued)
	if !js.firstGrant.IsZero() {
		waited = js.firstGrant.Sub(js.job.Enqueued)
	}
	out := JobState{
		ID:         js.job.ID,
		Class:      string(js.job.Class),
		Kind:       string(js.job.Kind),
		Model:      js.job.ModelID,
		State:      string(js.job.State),
		Enqueued:   js.job.Enqueued,
		WaitedMS:   waited.Milliseconds(),
		Steps:      js.job.Steps,
		Dispatched: js.dispatched,
		Completed:  js.completed,
		MaxWaitMS:  js.job.MaxWait.Milliseconds(),
	}
	if js.dispatched < js.job.Steps {
		out.Reason = js.reason
	}
	return out
}

func sortJobStates(l []JobState) {
	sort.Slice(l, func(i, j int) bool { return l[i].Enqueued.Before(l[j].Enqueued) })
}

func sortStrings(l []string) { sort.Strings(l) }
