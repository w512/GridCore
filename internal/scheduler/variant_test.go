package scheduler

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/job"
)

// Budget 15488 (16000 - 512 headroom). big + other do not fit together.
const familyModels = "" +
	"  big:   { runtime: sim, capabilities: [chat], simulated_vram_mb: 8000, simulated_load_time: 5ms, parallel: 2 }\n" +
	"  small: { runtime: sim, capabilities: [chat], simulated_vram_mb: 3000, simulated_load_time: 5ms, parallel: 2 }\n" +
	"  other: { runtime: sim, capabilities: [chat], simulated_vram_mb: 9000, simulated_load_time: 5ms, parallel: 1 }\n"

func (h *harness) submitFamily(c job.Class, variants ...string) *Handle {
	h.t.Helper()
	id := fmt.Sprintf("%s-%d", c, h.nextID.Add(1))
	j := job.NewFamily(id, context.Background(), c, job.Chat, "fam", variants)
	j.Enqueued = h.clock.Now()
	hd, err := h.s.Submit(j)
	if err != nil {
		h.t.Fatalf("submit %s: %v", id, err)
	}
	return hd
}

func (h *harness) variantEvent(jobID string) string {
	for _, e := range h.events(EvVariant) {
		if e.Subject == jobID {
			return e.Detail
		}
	}
	return ""
}

func TestFamilyIdleGPUGetsBestVariant(t *testing.T) {
	h := newHarness(t, 16000, familyModels)
	hd := h.submitFamily(job.Background, "big", "small")
	g := h.grant(hd, wait)
	if g.Model != "big" {
		t.Errorf("idle GPU: got %s, want big", g.Model)
	}
	hd.StepDone(g.Step, nil, Usage{})
	if d := h.variantEvent(hd.job.ID); d != "fam -> big (fits in free VRAM)" {
		t.Errorf("variant event = %q", d)
	}
}

// Background takes the variant that fits next to what is loaded rather
// than evict a model, even one it would be allowed to evict.
func TestFamilyBackgroundPrefersNoEviction(t *testing.T) {
	h := newHarness(t, 16000, familyModels)
	h.run(job.Background, "other")
	h.clock.Advance(time.Minute) // other is cold and past min_residency: evictable

	hd := h.submitFamily(job.Background, "big", "small")
	g := h.grant(hd, wait)
	hd.StepDone(g.Step, nil, Usage{})
	if g.Model != "small" {
		t.Errorf("got %s, want small (big would need evicting other)", g.Model)
	}
	if n := len(h.events(EvEvict)); n != 0 {
		t.Errorf("nothing should have been evicted, got %d evictions", n)
	}
	if d := h.variantEvent(hd.job.ID); !strings.Contains(d, "big needs evicting") {
		t.Errorf("the event should say why big was not used: %q", d)
	}
}

func TestFamilyBackgroundEvictsWhenNothingFits(t *testing.T) {
	h := newHarness(t, 13000, familyModels) // budget 12488: other leaves 3488, small needs 3512
	h.run(job.Background, "other")
	h.clock.Advance(time.Minute)

	hd := h.submitFamily(job.Background, "big", "small")
	g := h.grant(hd, wait)
	hd.StepDone(g.Step, nil, Usage{})
	if g.Model != "big" {
		t.Errorf("got %s, want big: when an eviction is needed anyway, the best variant", g.Model)
	}
}

// Interactive work gets the best variant as if it had asked for it, even
// when a lesser one is already resident.
func TestFamilyInteractiveGetsBestVariant(t *testing.T) {
	h := newHarness(t, 16000, familyModels)
	h.run(job.Background, "other")
	h.run(job.Background, "small")
	h.clock.Advance(time.Minute)

	hd := h.submitFamily(job.Interactive, "big", "small")
	g := h.grant(hd, wait)
	hd.StepDone(g.Step, nil, Usage{})
	if g.Model != "big" {
		t.Errorf("got %s, want big", g.Model)
	}
	if h.resident("other") != nil {
		t.Error("other should have been evicted for big")
	}
}

// ...and falls back only when the best variant cannot be had at all: here
// a busy hot model holds the memory big would need.
func TestFamilyInteractiveFallsBack(t *testing.T) {
	h := newHarness(t, 16000, familyModels)
	busy := h.submit(job.Interactive, "other", 1)
	gb := h.grant(busy, wait) // other is hot and busy until the end of the test

	hd := h.submitFamily(job.Interactive, "big", "small")
	g := h.grant(hd, wait)
	hd.StepDone(g.Step, nil, Usage{})
	if g.Model != "small" {
		t.Errorf("got %s, want small", g.Model)
	}
	if d := h.variantEvent(hd.job.ID); !strings.Contains(d, "big cannot fit now") {
		t.Errorf("variant event = %q", d)
	}

	// A request that accepts only big waits, and says why.
	only := h.submitFamily(job.Interactive, "big")
	h.noGrant(only, 50*time.Millisecond)
	h.eventually(func(st State) bool {
		for _, j := range st.Queued {
			if j.ID == only.job.ID {
				return j.Model == "fam" && j.Family == "fam" && strings.Contains(j.Reason, "no variant fits")
			}
		}
		return false
	}, "queued family job shows the family and the reason")

	busy.StepDone(gb.Step, nil, Usage{})
	_ = h.finish(busy, wait)
	g = h.grant(only, wait) // other is idle now and may be evicted for interactive
	only.StepDone(g.Step, nil, Usage{})
	if g.Model != "big" {
		t.Errorf("got %s, want big", g.Model)
	}
}

// Once a variant is picked the job sticks to it.
func TestFamilyChoiceIsFixed(t *testing.T) {
	h := newHarness(t, 16000, familyModels)
	hd := h.submitFamily(job.Background, "big", "small")
	g := h.grant(hd, wait)
	hd.StepDone(g.Step, nil, Usage{})
	if err := h.finish(hd, wait); err != nil {
		t.Fatal(err)
	}
	st := h.s.State()
	for _, e := range st.Events {
		if e.Kind == EvComplete && e.Subject == hd.job.ID && !strings.Contains(e.Detail, "big") {
			t.Errorf("complete event should name the variant: %q", e.Detail)
		}
	}
	if n := len(h.events(EvVariant)); n != 1 {
		t.Errorf("one variant decision per job, got %d", n)
	}
}

// Background does not take a slot of the model the user is chatting with
// when a lesser variant runs without disturbing anything.
func TestFamilyBackgroundAvoidsHotVariant(t *testing.T) {
	h := newHarness(t, 16000, familyModels)
	h.run(job.Interactive, "big") // big is hot, idle, with free slots
	h.clock.Advance(50 * time.Millisecond)

	hd := h.submitFamily(job.Background, "big", "small")
	g := h.grant(hd, wait)
	hd.StepDone(g.Step, nil, Usage{})
	if g.Model != "small" {
		t.Errorf("got %s, want small (big is in interactive use)", g.Model)
	}
	if d := h.variantEvent(hd.job.ID); !strings.Contains(d, "big resident, in interactive use") {
		t.Errorf("variant event = %q", d)
	}

	// Once big cools down it is just a resident model with a free slot.
	h.clock.Advance(time.Second) // hot_ttl in the harness is 200ms
	hd = h.submitFamily(job.Background, "big", "small")
	g = h.grant(hd, wait)
	hd.StepDone(g.Step, nil, Usage{})
	if g.Model != "big" {
		t.Errorf("got %s, want big once it is cold", g.Model)
	}
}
