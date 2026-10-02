package scheduler

import (
	"strings"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/job"
)

// Two models that do not fit together: 5000 + 5000 + headroom > 9488.
const (
	modelA = "  a: { runtime: sim, capabilities: [chat], simulated_vram_mb: 5000, simulated_load_time: 5ms, parallel: 2 }\n"
	modelB = "  b: { runtime: sim, capabilities: [chat], simulated_vram_mb: 5000, simulated_load_time: 5ms, parallel: 2 }\n"
)

func (h *harness) run(c job.Class, modelID string) {
	h.t.Helper()
	if err := h.runToCompletion(h.submit(c, modelID, 1), wait); err != nil {
		h.t.Fatal(err)
	}
}

func (h *harness) loadsOf(id string) int {
	n := 0
	for _, e := range h.events(EvLoad) {
		if e.Subject == id {
			n++
		}
	}
	return n
}

// The load-test thrash: background works on one model, batch wants another
// that does not fit next to it. In 0.1 every gap in the background stream
// let batch evict the background model, which came straight back.
func TestBatchWaitsWhileBackgroundUsesItsModel(t *testing.T) {
	h := newHarness(t, 10000, modelA+modelB)
	h.run(job.Background, "a")

	batch := h.submit(job.Batch, "b", 1)
	h.noGrant(batch, 50*time.Millisecond)
	h.eventually(func(st State) bool {
		return len(st.Queued) == 1 && strings.Contains(st.Queued[0].Reason, "a in use by background")
	}, "batch says which model it is waiting for")

	// Background keeps using a: batch keeps waiting, a is never reloaded.
	for i := 0; i < 5; i++ {
		h.clock.Advance(10 * time.Second)
		h.run(job.Background, "a")
	}
	h.noGrant(batch, 20*time.Millisecond)
	if n := h.loadsOf("a"); n != 1 {
		t.Errorf("a loaded %d times, want 1", n)
	}

	// Background goes quiet for min_residency (30s): batch gets the GPU.
	h.clock.Advance(31 * time.Second)
	if err := h.runToCompletion(batch, wait); err != nil {
		t.Fatal(err)
	}
	// And background takes it back at once: batch's model is not protected
	// from a higher class.
	h.run(job.Background, "a")
	if n := h.loadsOf("a"); n != 2 {
		t.Errorf("a loaded %d times, want 2", n)
	}
}

// Two background clients on two models that do not fit: each model gets
// min_residency before the other replaces it, instead of a reload per step.
func TestSameClassTakesTurnsAfterMinResidency(t *testing.T) {
	h := newHarness(t, 10000, modelA+modelB)
	h.run(job.Background, "a")
	h.clock.Advance(5 * time.Second)

	other := h.submit(job.Background, "b", 1)
	h.noGrant(other, 50*time.Millisecond)
	h.eventually(func(st State) bool {
		return len(st.Queued) == 1 && strings.Contains(st.Queued[0].Reason, "a loaded 5s ago")
	}, "the second model waits for the first one's turn")

	h.clock.Advance(26 * time.Second) // a resident for 31s
	if err := h.runToCompletion(other, wait); err != nil {
		t.Fatal(err)
	}
	if h.resident("a") != nil {
		t.Error("a should have been evicted for b")
	}
}

func TestInteractiveIgnoresMinResidency(t *testing.T) {
	h := newHarness(t, 10000, modelA+modelB)
	h.run(job.Background, "a")
	h.run(job.Interactive, "b") // a was loaded a moment ago; interactive does not wait
	if h.resident("a") != nil {
		t.Error("interactive work must be able to evict a freshly loaded background model")
	}
}

func TestMinResidencyZeroRestoresImmediateEviction(t *testing.T) {
	h := newHarnessWith(t, 10000, modelA+modelB, "  min_residency: 0s\n  eviction: lru\n")
	h.run(job.Background, "a")
	h.run(job.Batch, "b")
	h.run(job.Background, "a")
	if n := h.loadsOf("a"); n != 2 {
		t.Errorf("a loaded %d times, want 2 (0.1 behaviour)", n)
	}
}

func TestThrashIsReported(t *testing.T) {
	h := newHarnessWith(t, 10000, modelA+modelB, "  min_residency: 0s\n")
	for i := 0; i < thrashLoads; i++ {
		h.run(job.Background, "a")
		h.run(job.Background, "b")
		h.clock.Advance(time.Second)
	}
	ev := h.events(EvThrash)
	if len(ev) == 0 {
		t.Fatalf("no thrash event after %d reloads of each model\n%s", thrashLoads, h.dump())
	}
	if !strings.Contains(ev[0].Detail, "do not fit together") {
		t.Errorf("thrash detail = %q", ev[0].Detail)
	}
	// Once per window per model, not on every further load.
	h.run(job.Background, "a")
	h.run(job.Background, "b")
	if n := len(h.events(EvThrash)); n != len(ev) {
		t.Errorf("thrash reported again within the window: %d -> %d", len(ev), n)
	}
}

// With demand recorded, the model in steady use survives even though the
// other one was touched more recently.
func TestCostEvictionKeepsModelInDemand(t *testing.T) {
	models := modelA + modelB +
		"  c: { runtime: sim, capabilities: [chat], simulated_vram_mb: 5000, simulated_load_time: 5ms }\n"
	h := newHarness(t, 16000, models) // a and b fit together, c needs one of them gone
	for i := 0; i < 10; i++ {
		h.run(job.Background, "a")
		h.clock.Advance(2 * time.Second)
	}
	h.run(job.Background, "b") // used once, but most recently
	h.clock.Advance(time.Minute)

	h.run(job.Interactive, "c")
	if h.resident("a") == nil {
		t.Errorf("a (steady demand) should stay; LRU would have evicted it\n%s", h.dump())
	}
	if h.resident("b") != nil {
		t.Errorf("b (one request) should have been evicted\n%s", h.dump())
	}
	for _, e := range h.events(EvEvict) {
		if !strings.Contains(e.Detail, "cost ") {
			t.Errorf("evict event should state the cost: %q", e.Detail)
		}
	}
}

// Batch must not starve forever behind a background stream that keeps its
// model busy: after batch_max_starvation (2m) it takes the model, once.
func TestBatchTakesBackgroundModelWhenOverdue(t *testing.T) {
	h := newHarness(t, 10000, modelA+modelB)
	h.run(job.Background, "a")
	batch := h.submit(job.Batch, "b", 1)
	for i := 0; i < 11; i++ { // 110s of steady background use
		h.clock.Advance(10 * time.Second)
		h.run(job.Background, "a")
	}
	h.noGrant(batch, 30*time.Millisecond)

	h.clock.Advance(15 * time.Second) // 125s since the batch job arrived
	if err := h.runToCompletion(batch, wait); err != nil {
		t.Fatal(err)
	}
	if n := h.loadsOf("b"); n != 1 {
		t.Errorf("b loaded %d times", n)
	}
	// Background takes its model back right away; the next batch job waits
	// its turn again instead of ping-ponging.
	h.run(job.Background, "a")
	next := h.submit(job.Batch, "b", 1)
	h.clock.Advance(10 * time.Second)
	h.run(job.Background, "a")
	h.noGrant(next, 30*time.Millisecond)
	if n := h.loadsOf("a"); n != 2 {
		t.Errorf("a loaded %d times, want 2", n)
	}
}

// An overdue batch job also runs while background work is queued.
func TestOverdueBatchRunsPastQueuedBackground(t *testing.T) {
	models := "  a: { runtime: sim, capabilities: [chat], simulated_vram_mb: 3000, simulated_load_time: 5ms, parallel: 1 }\n" + modelB
	h := newHarness(t, 16000, models)
	busy := h.submit(job.Background, "a", 1)
	g := h.grant(busy, wait)                   // a's only slot is taken...
	queued := h.submit(job.Background, "a", 1) // ...so this one stays queued
	batch := h.submit(job.Batch, "b", 1)
	h.noGrant(batch, 40*time.Millisecond)
	h.eventually(func(st State) bool {
		for _, j := range st.Queued {
			if j.ID == batch.job.ID {
				return j.Reason == "background queued"
			}
		}
		return false
	}, "batch waits behind queued background work")

	h.clock.Advance(2*time.Minute + time.Second)
	gb := h.grant(batch, wait)
	batch.StepDone(gb.Step, nil, Usage{})
	found := false
	for _, e := range h.events(EvPreempt) {
		found = found || strings.Contains(e.Detail, "behind background")
	}
	if !found {
		t.Errorf("expected a 'waited behind background' event: %v", h.events(EvPreempt))
	}
	busy.StepDone(g.Step, nil, Usage{})
	_ = h.runToCompletion(queued, wait)
}

func TestBatchMaxStarvationZeroKeepsStrictOrder(t *testing.T) {
	h := newHarnessWith(t, 10000, modelA+modelB, "  batch_max_starvation: 0s\n")
	h.run(job.Background, "a")
	batch := h.submit(job.Batch, "b", 1)
	for i := 0; i < 20; i++ {
		h.clock.Advance(10 * time.Second)
		h.run(job.Background, "a")
	}
	h.noGrant(batch, 30*time.Millisecond)
}
