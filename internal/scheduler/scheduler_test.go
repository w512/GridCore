package scheduler

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gridcore/gridcore/internal/config"
	"github.com/gridcore/gridcore/internal/gpu"
	gpufake "github.com/gridcore/gridcore/internal/gpu/fake"
	"github.com/gridcore/gridcore/internal/job"
	"github.com/gridcore/gridcore/internal/runtime"
	rtfake "github.com/gridcore/gridcore/internal/runtime/fake"
)

const (
	chat   = "  chat:   { runtime: sim, capabilities: [chat], fake_vram_mb: 9000, fake_load_time: 20ms, parallel: 1 }\n"
	chat2  = "  chat2:  { runtime: sim, capabilities: [chat], fake_vram_mb: 9000, fake_load_time: 20ms, parallel: 1 }\n"
	vision = "  vision: { runtime: sim, capabilities: [chat, vision], fake_vram_mb: 6000, fake_load_time: 20ms, parallel: 1 }\n"
	embed  = "  embed:  { runtime: sim, capabilities: [embedding], fake_vram_mb: 600, fake_load_time: 10ms, parallel: 4 }\n"
	pinned = "  embed:  { runtime: sim, capabilities: [embedding], fake_vram_mb: 600, fake_load_time: 10ms, parallel: 4, pinned: true }\n"
)

const wait = 3 * time.Second

func TestPreloadPinned(t *testing.T) {
	h := newHarness(t, 16000, chat+pinned)
	h.eventually(isReady("embed"), "pinned model preloaded")
	if h.resident("chat") != nil {
		t.Error("non-preload model must not be loaded at start")
	}
	if r := h.resident("embed"); r.Tier != "pinned" {
		t.Errorf("tier = %s", r.Tier)
	}
}

func TestInteractiveFirstStrictOrder(t *testing.T) {
	h := newHarness(t, 16000, chat)
	// Same model, one slot: order of grants shows class ordering.
	batch := h.submit(job.Batch, "chat", 1)
	bg := h.submit(job.Background, "chat", 1)
	inter := h.submit(job.Interactive, "chat", 1)

	g := h.grant(inter, wait)
	h.noGrant(bg, 50*time.Millisecond)
	h.noGrant(batch, 10*time.Millisecond)
	inter.StepDone(g.Step, nil, Usage{})
	if err := h.finish(inter, wait); err != nil {
		t.Fatal(err)
	}

	// Idle window must pass before background may start.
	h.noGrant(bg, 20*time.Millisecond)
	h.clock.Advance(50 * time.Millisecond)
	g = h.grant(bg, wait)
	h.noGrant(batch, 20*time.Millisecond) // background still queued? no: it is dispatched; batch waits for the slot
	bg.StepDone(g.Step, nil, Usage{})
	if err := h.finish(bg, wait); err != nil {
		t.Fatal(err)
	}
	g = h.grant(batch, wait)
	batch.StepDone(g.Step, nil, Usage{})
	if err := h.finish(batch, wait); err != nil {
		t.Fatal(err)
	}
}

func TestBackgroundYieldsBetweenSteps(t *testing.T) {
	h := newHarness(t, 16000, chat+"  embed:  { runtime: sim, capabilities: [embedding], fake_vram_mb: 600, fake_load_time: 10ms, parallel: 1 }\n")
	bg := h.submit(job.Background, "embed", 10)

	g0 := h.grant(bg, wait)
	bg.StepDone(g0.Step, nil, Usage{})
	g1 := h.grant(bg, wait)
	if g1.Step != 1 {
		t.Fatalf("step = %d", g1.Step)
	}

	// Interactive arrives while step 1 is running.
	inter := h.submit(job.Interactive, "chat", 1)
	bg.StepDone(g1.Step, nil, Usage{})
	gi := h.grant(inter, wait)

	// No background step may start while the interactive one runs.
	h.noGrant(bg, 60*time.Millisecond)
	if st := h.s.State(); st.Mode != "interactive" {
		t.Errorf("mode = %s", st.Mode)
	}
	inter.StepDone(gi.Step, nil, Usage{})
	if err := h.finish(inter, wait); err != nil {
		t.Fatal(err)
	}
	// Still inside the idle window.
	h.noGrant(bg, 20*time.Millisecond)
	h.clock.Advance(50 * time.Millisecond)
	g2 := h.grant(bg, wait)
	if g2.Step != 2 {
		t.Fatalf("resumed at step %d, want 2", g2.Step)
	}
	bg.StepDone(g2.Step, nil, Usage{})
	if err := h.runToCompletion(bg, wait); err != nil {
		t.Fatal(err)
	}
	if h.rt.Instance("embed").Requests() != 0 {
		t.Log("note: harness completes steps without HTTP; requests counter stays 0")
	}
}

func TestColdEvictedForInteractive(t *testing.T) {
	// budget 11488: vision (6000+512) + chat (9000+512) do not fit together.
	h := newHarness(t, 12000, chat+vision)
	if err := h.s.LoadModel("vision"); err != nil {
		t.Fatal(err)
	}
	h.eventually(isReady("vision"), "vision loaded")

	inter := h.submit(job.Interactive, "chat", 1)
	g := h.grant(inter, wait)
	if g.Model != "chat" {
		t.Fatalf("grant model = %s", g.Model)
	}
	h.eventually(notResident("vision"), "vision evicted")
	inter.StepDone(g.Step, nil, Usage{})
	if err := h.finish(inter, wait); err != nil {
		t.Fatal(err)
	}
	// Order of events: evict vision -> unloaded -> load chat -> loaded.
	var order []string
	for _, e := range h.s.State().Events {
		switch e.Kind {
		case EvEvict, EvUnloaded, EvLoad, EvLoaded:
			order = append(order, e.Kind+":"+e.Subject)
		}
	}
	want := []string{"load:vision", "loaded:vision", "evict:vision", "unloaded:vision", "load:chat", "loaded:chat"}
	if len(order) != len(want) {
		t.Fatalf("events = %v", order)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("events = %v, want %v", order, want)
		}
	}
}

func TestHotNotEvictedForBackground(t *testing.T) {
	h := newHarness(t, 12000, chat+vision)
	inter := h.submit(job.Interactive, "chat", 1)
	if err := h.runToCompletion(inter, wait); err != nil {
		t.Fatal(err)
	}
	h.clock.Advance(50 * time.Millisecond) // past idle window, inside hot_ttl (200ms)

	bg := h.submit(job.Background, "vision", 1)
	h.noGrant(bg, 60*time.Millisecond)
	if r := h.resident("chat"); r == nil || r.State != "ready" || r.Tier != "hot" {
		t.Fatalf("chat must stay resident and hot: %+v", r)
	}
	h.eventually(func(st State) bool {
		return len(st.Queued) == 1 && st.Queued[0].Reason == "waiting for VRAM"
	}, "background waits for VRAM")

	// Once chat cools down it becomes a cold LRU victim.
	h.clock.Advance(200 * time.Millisecond)
	g := h.grant(bg, wait)
	if g.Model != "vision" {
		t.Fatalf("grant = %+v", g)
	}
	h.eventually(notResident("chat"), "cold chat evicted for background")
	bg.StepDone(g.Step, nil, Usage{})
	_ = h.finish(bg, wait)
}

func TestHotEvictedForInteractiveOnlyWhenIdle(t *testing.T) {
	h := newHarness(t, 12000, chat+vision)
	a := h.submit(job.Interactive, "chat", 1)
	ga := h.grant(a, wait)

	// Second interactive model does not fit while chat is hot and busy.
	b := h.submit(job.Interactive, "vision", 1)
	h.noGrant(b, 60*time.Millisecond)
	if r := h.resident("chat"); r == nil || r.State != "ready" {
		t.Fatalf("busy hot chat must not be evicted: %+v", r)
	}

	// As soon as chat is idle it may be evicted for interactive work.
	a.StepDone(ga.Step, nil, Usage{})
	if err := h.finish(a, wait); err != nil {
		t.Fatal(err)
	}
	gb := h.grant(b, wait)
	if gb.Model != "vision" {
		t.Fatalf("grant = %+v", gb)
	}
	h.eventually(notResident("chat"), "hot idle chat evicted for interactive")
	b.StepDone(gb.Step, nil, Usage{})
	_ = h.finish(b, wait)
}

func TestPinnedNeverEvictedAndTooLargeFailsFast(t *testing.T) {
	// budget 9988; pinned embed 600 leaves 9388 < chat need 9512.
	h := newHarness(t, 10500, chat+pinned)
	h.eventually(isReady("embed"), "pinned loaded")

	inter := h.submit(job.Interactive, "chat", 1)
	err := h.finish(inter, wait)
	if !errors.Is(err, ErrModelTooLarge) {
		t.Fatalf("want ErrModelTooLarge, got %v", err)
	}
	if r := h.resident("embed"); r == nil || r.State != "ready" {
		t.Fatalf("pinned must remain resident: %+v", r)
	}
}

func TestPinnedSurvivesEvictionOfCold(t *testing.T) {
	// budget 15488: embed 600 + vision 6512 + chat 9512 = 16624 > budget.
	h := newHarness(t, 16000, chat+vision+pinned)
	h.eventually(isReady("embed"), "pinned loaded")
	if err := h.s.LoadModel("vision"); err != nil {
		t.Fatal(err)
	}
	h.eventually(isReady("vision"), "vision loaded")

	inter := h.submit(job.Interactive, "chat", 1)
	g := h.grant(inter, wait)
	h.eventually(notResident("vision"), "cold vision evicted")
	if r := h.resident("embed"); r == nil || r.State != "ready" {
		t.Fatalf("pinned must remain: %+v", r)
	}
	inter.StepDone(g.Step, nil, Usage{})
	_ = h.finish(inter, wait)
}

func TestDrainBeforeStop(t *testing.T) {
	h := newHarness(t, 12000, chat+vision)
	bg := h.submit(job.Background, "vision", 1)
	gv := h.grant(bg, wait) // vision running a background step

	inter := h.submit(job.Interactive, "chat", 1)
	h.eventually(func(st State) bool {
		for _, r := range st.Resident {
			if r.ID == "vision" && r.State == "draining" {
				return true
			}
		}
		return false
	}, "vision marked draining")
	h.noGrant(inter, 50*time.Millisecond)
	if r := h.resident("vision"); r == nil || r.BusySlots != 1 {
		t.Fatalf("draining instance must not be stopped while busy: %+v", r)
	}

	bg.StepDone(gv.Step, nil, Usage{})
	if err := h.finish(bg, wait); err != nil {
		t.Fatal(err)
	}
	g := h.grant(inter, wait)
	h.eventually(notResident("vision"), "vision stopped after drain")
	inter.StepDone(g.Step, nil, Usage{})
	_ = h.finish(inter, wait)
}

func TestMaxWaitTimeout(t *testing.T) {
	h := newHarness(t, 16000, "  slow: { runtime: sim, capabilities: [chat], fake_vram_mb: 9000, fake_load_time: 500ms }\n")
	inter := h.submit(job.Interactive, "slow", 1, withMaxWait(20*time.Millisecond))
	h.clock.Advance(25 * time.Millisecond)
	err := h.finish(inter, wait)
	if !errors.Is(err, ErrQueueTimeout) {
		t.Fatalf("want ErrQueueTimeout, got %v", err)
	}
	if st := h.s.State(); len(st.Queued) != 0 {
		t.Errorf("queue should be empty: %+v", st.Queued)
	}
	if len(h.events(EvTimeout)) != 1 {
		t.Error("expected one timeout event")
	}
}

func TestCancelQueuedAndRunning(t *testing.T) {
	h := newHarness(t, 16000, chat)
	ctx, cancel := context.WithCancel(context.Background())
	a := h.submit(job.Interactive, "chat", 1, withCtx(ctx))
	ga := h.grant(a, wait)

	// b queued behind a (one slot); cancel while queued.
	ctxB, cancelB := context.WithCancel(context.Background())
	b := h.submit(job.Interactive, "chat", 1, withCtx(ctxB))
	h.noGrant(b, 20*time.Millisecond)
	cancelB()
	if err := h.finish(b, wait); !errors.Is(err, ErrCancelled) {
		t.Fatalf("queued cancel: %v", err)
	}

	// Cancel a while running: handle finishes, slot is returned on StepDone.
	cancel()
	if err := h.finish(a, wait); !errors.Is(err, ErrCancelled) {
		t.Fatalf("running cancel: %v", err)
	}
	if r := h.resident("chat"); r.BusySlots != 1 {
		t.Fatalf("slot must be held until StepDone: %+v", r)
	}
	a.StepDone(ga.Step, errors.New("aborted"), Usage{})
	h.eventually(func(st State) bool {
		for _, r := range st.Resident {
			if r.ID == "chat" {
				return r.BusySlots == 0
			}
		}
		return false
	}, "slot returned")
	if st := h.s.State(); len(st.Running) != 0 {
		t.Errorf("running should be empty: %+v", st.Running)
	}
}

func TestInstanceCrash(t *testing.T) {
	h := newHarness(t, 16000, chat)
	a := h.submit(job.Interactive, "chat", 1)
	g := h.grant(a, wait)

	h.rt.Instance("chat").Kill()
	if err := h.finish(a, wait); !errors.Is(err, ErrInstanceFailed) {
		t.Fatalf("want ErrInstanceFailed, got %v", err)
	}
	h.eventually(notResident("chat"), "crashed instance removed")
	a.StepDone(g.Step, errors.New("connection refused"), Usage{}) // late report must be harmless

	// Next request reloads the model.
	b := h.submit(job.Interactive, "chat", 1)
	if err := h.runToCompletion(b, wait); err != nil {
		t.Fatal(err)
	}
	if len(h.events(EvCrash)) != 1 {
		t.Errorf("crash events = %d", len(h.events(EvCrash)))
	}
}

func TestCircuitBreaker(t *testing.T) {
	h := newHarness(t, 16000, chat)
	h.rt.SetFailLoad(errors.New("boom"))

	for i := 0; i < 3; i++ {
		j := h.submit(job.Interactive, "chat", 1)
		if err := h.finish(j, wait); !errors.Is(err, ErrLoadFailed) {
			t.Fatalf("attempt %d: want ErrLoadFailed, got %v", i, err)
		}
	}
	h.eventually(func(st State) bool { return len(st.Disabled) == 1 && st.Disabled[0] == "chat" }, "chat disabled")

	j := h.submit(job.Interactive, "chat", 1)
	if err := h.finish(j, wait); !errors.Is(err, ErrModelDisabled) {
		t.Fatalf("want ErrModelDisabled, got %v", err)
	}

	h.rt.SetFailLoad(nil)
	if err := h.s.EnableModel("chat"); err != nil {
		t.Fatal(err)
	}
	j = h.submit(job.Interactive, "chat", 1)
	if err := h.runToCompletion(j, wait); err != nil {
		t.Fatalf("after enable: %v", err)
	}
}

func TestReservationsPreventOvercommit(t *testing.T) {
	// budget 15488; two 9512 models never fit together.
	h := newHarness(t, 16000, chat+chat2)
	a := h.submit(job.Interactive, "chat", 1)
	b := h.submit(job.Interactive, "chat2", 1)

	stop := make(chan struct{})
	violations := make(chan string, 1)
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
			}
			st := h.s.State()
			if st.GPU.CommittedMB > st.GPU.BudgetMB {
				select {
				case violations <- h.dump():
				default:
				}
				return
			}
			time.Sleep(time.Millisecond)
		}
	}()

	ga := h.grant(a, wait)
	h.noGrant(b, 60*time.Millisecond)
	a.StepDone(ga.Step, nil, Usage{})
	_ = h.finish(a, wait)
	gb := h.grant(b, wait)
	b.StepDone(gb.Step, nil, Usage{})
	_ = h.finish(b, wait)
	close(stop)
	select {
	case v := <-violations:
		t.Fatalf("committed exceeded budget: %s", v)
	default:
	}
}

func TestParallelSlotsRespected(t *testing.T) {
	h := newHarness(t, 16000, "  chat: { runtime: sim, capabilities: [chat], fake_vram_mb: 9000, fake_load_time: 10ms, parallel: 2 }\n")
	var hs []*Handle
	for i := 0; i < 5; i++ {
		hs = append(hs, h.submit(job.Interactive, "chat", 1))
	}
	g0 := h.grant(hs[0], wait)
	g1 := h.grant(hs[1], wait)
	h.noGrant(hs[2], 40*time.Millisecond)
	if r := h.resident("chat"); r.BusySlots != 2 {
		t.Fatalf("busy = %d", r.BusySlots)
	}
	hs[0].StepDone(g0.Step, nil, Usage{})
	g2 := h.grant(hs[2], wait)
	h.noGrant(hs[3], 20*time.Millisecond)
	hs[1].StepDone(g1.Step, nil, Usage{})
	hs[2].StepDone(g2.Step, nil, Usage{})
	for _, hd := range hs[3:] {
		if err := h.runToCompletion(hd, wait); err != nil {
			t.Fatal(err)
		}
	}
	for _, hd := range hs[:3] {
		_ = h.finish(hd, wait)
	}
}

func TestMultiStepUsesAllSlots(t *testing.T) {
	h := newHarness(t, 16000, embed)
	bg := h.submit(job.Background, "embed", 6)
	var grants []Grant
	for i := 0; i < 4; i++ {
		grants = append(grants, h.grant(bg, wait))
	}
	h.noGrant(bg, 20*time.Millisecond)
	for _, g := range grants {
		bg.StepDone(g.Step, nil, Usage{})
	}
	if err := h.runToCompletion(bg, wait); err != nil {
		t.Fatal(err)
	}
}

func TestStepErrorFailsJob(t *testing.T) {
	h := newHarness(t, 16000, embed)
	bg := h.submit(job.Background, "embed", 3)
	g := h.grant(bg, wait)
	bg.StepDone(g.Step, errors.New("upstream 500"), Usage{})
	err := h.finish(bg, wait)
	if !errors.Is(err, ErrStepFailed) {
		t.Fatalf("want ErrStepFailed, got %v", err)
	}
	// Remaining grants that were already issued may still be reported.
	for {
		select {
		case g := <-bg.Grants():
			bg.StepDone(g.Step, errors.New("abandoned"), Usage{})
			continue
		default:
		}
		break
	}
	h.eventually(func(st State) bool {
		for _, r := range st.Resident {
			if r.ID == "embed" {
				return r.BusySlots == 0
			}
		}
		return false
	}, "slots returned after failure")
}

func TestBatchWaitsForBackground(t *testing.T) {
	h := newHarness(t, 16000, chat)
	bg := h.submit(job.Background, "chat", 1)
	batch := h.submit(job.Batch, "chat", 1)
	g := h.grant(bg, wait)
	h.eventually(func(st State) bool {
		return len(st.Queued) == 1 && st.Queued[0].Reason == "waiting for slot"
	}, "batch waits for the slot")
	bg.StepDone(g.Step, nil, Usage{})
	_ = h.finish(bg, wait)
	g = h.grant(batch, wait)
	batch.StepDone(g.Step, nil, Usage{})
	_ = h.finish(batch, wait)
}

func TestAdminLoadUnload(t *testing.T) {
	h := newHarness(t, 16000, chat+pinned)
	h.eventually(isReady("embed"), "pinned loaded")
	if err := h.s.UnloadModel("embed"); err == nil {
		t.Fatal("unloading a pinned model must fail")
	}
	if err := h.s.UnloadModel("chat"); err == nil {
		t.Fatal("unloading a non-resident model must fail")
	}
	if err := h.s.LoadModel("chat"); err != nil {
		t.Fatal(err)
	}
	h.eventually(isReady("chat"), "chat loaded by admin")
	if err := h.s.UnloadModel("chat"); err != nil {
		t.Fatal(err)
	}
	h.eventually(notResident("chat"), "chat unloaded by admin")
	if err := h.s.LoadModel("nope"); err == nil {
		t.Fatal("unknown model must fail")
	}
}

func TestSubmitValidation(t *testing.T) {
	h := newHarness(t, 16000, chat)
	j := job.New("x", context.Background(), job.Interactive, job.Chat, "nope")
	if _, err := h.s.Submit(j); err == nil {
		t.Error("unknown model must be rejected")
	}
	j = job.New("y", context.Background(), "urgent", job.Chat, "chat")
	if _, err := h.s.Submit(j); err == nil {
		t.Error("unknown class must be rejected")
	}
	j = job.New("z", context.Background(), job.Interactive, job.Chat, "chat")
	j.Steps = 0
	if _, err := h.s.Submit(j); err == nil {
		t.Error("zero steps must be rejected")
	}
}

func TestShutdownFailsQueuedAndStopsInstances(t *testing.T) {
	h := newHarness(t, 16000, chat)
	a := h.submit(job.Interactive, "chat", 1)
	ga := h.grant(a, wait)
	b := h.submit(job.Interactive, "chat", 1) // queued behind a

	h.cancel()
	if err := h.finish(b, wait); !errors.Is(err, ErrShuttingDown) {
		t.Fatalf("queued job on shutdown: %v", err)
	}
	// In-flight step completes normally during the drain window.
	a.StepDone(ga.Step, nil, Usage{})
	if err := h.finish(a, wait); err != nil {
		t.Fatalf("in-flight job should complete: %v", err)
	}
	select {
	case <-h.done:
	case <-time.After(wait):
		t.Fatal("Run did not return")
	}
	if inst := h.rt.Instance("chat"); inst != nil {
		t.Error("instance should be stopped after shutdown")
	}
	if _, err := h.s.Submit(job.New("late", context.Background(), job.Interactive, job.Chat, "chat")); !errors.Is(err, ErrShuttingDown) {
		t.Errorf("submit after stop: %v", err)
	}
}

func TestEvictsOnlyAsMuchAsNeeded(t *testing.T) {
	// Reproduces a bug seen on real hardware: two 7 GB models resident, a
	// 1 GB model requested. One victim frees plenty, but while it drained
	// the planner re-ran, saw no free memory yet, and evicted the second
	// model too. Memory of draining instances must count as pending.
	models := "  a: { runtime: sim, capabilities: [chat], fake_vram_mb: 7000, fake_load_time: 10ms }\n" +
		"  b: { runtime: sim, capabilities: [chat], fake_vram_mb: 7000, fake_load_time: 10ms }\n" +
		"  c: { runtime: sim, capabilities: [chat], fake_vram_mb: 1000, fake_load_time: 10ms }\n"
	h := newHarness(t, 16000, models)
	h.rt.StopDelay = 200 * time.Millisecond // teardown takes many ticks
	for _, id := range []string{"a", "b"} {
		if err := h.s.LoadModel(id); err != nil {
			t.Fatal(err)
		}
		h.eventually(isReady(id), id+" loaded")
		h.clock.Advance(time.Second) // a is LRU
	}
	// budget 15488 - 7512 - 7512 = 464 free; c needs 1512 -> evict exactly one.
	inter := h.submit(job.Interactive, "c", 1)
	time.Sleep(80 * time.Millisecond) // ~16 ticks while a drains
	st := h.s.State()
	var evicting, ready []string
	for _, r := range st.Resident {
		switch r.State {
		case "draining", "stopping":
			evicting = append(evicting, r.ID)
		case "ready":
			ready = append(ready, r.ID)
		}
	}
	if len(evicting) != 1 || evicting[0] != "a" || len(ready) != 1 || ready[0] != "b" {
		t.Fatalf("expected only a evicting and b ready, got evicting=%v ready=%v\n%s", evicting, ready, h.dump())
	}
	g := h.grant(inter, wait)
	inter.StepDone(g.Step, nil, Usage{})
	if err := h.finish(inter, wait); err != nil {
		t.Fatal(err)
	}
	if r := h.resident("b"); r == nil || r.State != "ready" {
		t.Fatalf("b must survive: %+v", r)
	}
	if n := len(h.events(EvEvict)); n != 1 {
		t.Errorf("evict events = %d, want 1", n)
	}
}

func TestOOMDiscardsProfile(t *testing.T) {
	h := newHarness(t, 16000, chat)
	// First load succeeds and measures 9000 MB into the profile.
	a := h.submit(job.Interactive, "chat", 1)
	if err := h.runToCompletion(a, wait); err != nil {
		t.Fatal(err)
	}
	h.eventually(func(st State) bool {
		for _, r := range st.Resident {
			if r.ID == "chat" && r.Measured {
				return true
			}
		}
		return false
	}, "chat VRAM measured by a gpu snapshot")
	if err := h.s.UnloadModel("chat"); err != nil {
		t.Fatal(err)
	}
	h.eventually(notResident("chat"), "unloaded")
	key := h.s.specs["chat"].ProfileKey("fake", "FakeGPU")
	if p, ok := h.s.store.Get(key); !ok || p.VRAMMB != 9000 {
		t.Fatalf("profile after first load = %+v ok=%v", p, ok)
	}

	// Next load hits OOM: the profile must be discarded, the job fails.
	h.rt.SetFailLoad(&rtfake.OOMError{Msg: "failed to allocate CUDA0 buffer"})
	b := h.submit(job.Interactive, "chat", 1)
	if err := h.finish(b, wait); !errors.Is(err, ErrLoadFailed) {
		t.Fatalf("want ErrLoadFailed, got %v", err)
	}
	if p, ok := h.s.store.Get(key); ok && p.VRAMMB != 0 {
		t.Fatalf("profile VRAM must be reset after OOM, got %d", p.VRAMMB)
	}
	found := false
	for _, e := range h.events(EvLoadFail) {
		if strings.Contains(e.Detail, "discarded") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected a 'profile discarded' event, got %+v", h.events(EvLoadFail))
	}

	// Recovery: load works again and the estimate (+headroom) is used.
	h.rt.SetFailLoad(nil)
	c := h.submit(job.Interactive, "chat", 1)
	if err := h.runToCompletion(c, wait); err != nil {
		t.Fatal(err)
	}
	for _, e := range h.s.State().Events {
		if e.Kind == EvLoad && strings.Contains(e.Detail, "reserve 9512") {
			return
		}
	}
	t.Errorf("after OOM the reservation should come from the estimate (9000+512), events: %v", h.events(EvLoad))
}

func TestBoundedStarvationUnderContinuousInteractive(t *testing.T) {
	h := newHarness(t, 16000, chat+embed)
	// Keep the GPU in interactive mode for the whole test: one interactive
	// job is always running (chat has one slot; we keep it busy).
	cur := h.submit(job.Interactive, "chat", 1)
	g := h.grant(cur, wait)

	bg := h.submit(job.Background, "embed", 6)
	// Interactive mode, no starvation yet -> nothing for background even
	// though embed could be loaded and run.
	h.noGrant(bg, 60*time.Millisecond)
	if st := h.s.State(); st.Mode != "interactive" {
		t.Fatalf("mode = %s", st.Mode)
	}

	// Past the starvation limit (3s default): exactly one step goes through.
	h.clock.Advance(3100 * time.Millisecond)
	g0 := h.grant(bg, wait)
	if g0.Step != 0 {
		t.Fatalf("step = %d", g0.Step)
	}
	h.noGrant(bg, 60*time.Millisecond) // one at a time while interactive is active
	if len(h.events(EvPreempt)) != 1 {
		t.Errorf("expected one starvation event, got %d", len(h.events(EvPreempt)))
	}

	// Finishing the step resets progress; the next one waits for the limit again.
	bg.StepDone(g0.Step, nil, Usage{})
	h.noGrant(bg, 60*time.Millisecond)
	h.clock.Advance(3100 * time.Millisecond)
	g1 := h.grant(bg, wait)
	if g1.Step != 1 {
		t.Fatalf("step = %d", g1.Step)
	}
	bg.StepDone(g1.Step, nil, Usage{})

	// Interactive goes idle: the remaining steps flow freely (all slots).
	cur.StepDone(g.Step, nil, Usage{})
	_ = h.finish(cur, wait)
	h.clock.Advance(100 * time.Millisecond) // past the 30ms idle window
	var got []int
	for i := 0; i < 4; i++ {
		got = append(got, h.grant(bg, wait).Step)
	}
	if len(got) != 4 {
		t.Fatalf("remaining steps should all be dispatched at once, got %v", got)
	}
	for _, s := range got {
		bg.StepDone(s, nil, Usage{})
	}
	if err := h.finish(bg, wait); err != nil {
		t.Fatal(err)
	}
}

func TestStarvationGuardDisabled(t *testing.T) {
	h := newHarnessWith(t, 16000, chat+embed, "  background_max_starvation: 0s\n")
	cur := h.submit(job.Interactive, "chat", 1)
	g := h.grant(cur, wait)
	bg := h.submit(job.Background, "embed", 2)
	h.clock.Advance(time.Minute)
	h.noGrant(bg, 60*time.Millisecond)
	cur.StepDone(g.Step, nil, Usage{})
	_ = h.finish(cur, wait)
	h.clock.Advance(100 * time.Millisecond)
	_ = h.runToCompletion(bg, wait)
}

func TestStarvationGuardSkipsUnloadableHead(t *testing.T) {
	// chat (9512) is hot and busy; vision (6512) cannot be loaded next to it
	// (budget 15488) and background may not evict hot models. A background
	// job on vision is the oldest starved job; an embed job whose model is
	// resident must still get its step.
	// Starvation limit (100ms) below hot_ttl (200ms) so chat stays hot.
	h := newHarnessWith(t, 16000, chat+vision+pinned, "  background_max_starvation: 100ms\n")
	h.eventually(isReady("embed"), "pinned loaded")
	cur := h.submit(job.Interactive, "chat", 1)
	g := h.grant(cur, wait)

	bgVision := h.submit(job.Background, "vision", 1)
	h.clock.Advance(10 * time.Millisecond)
	bgEmbed := h.submit(job.Background, "embed", 3)
	h.clock.Advance(150 * time.Millisecond)

	ge := h.grant(bgEmbed, wait)
	if ge.Model != "embed" {
		t.Fatalf("grant = %+v", ge)
	}
	h.noGrant(bgVision, 40*time.Millisecond)
	if r := h.resident("chat"); r == nil || r.State != "ready" {
		t.Fatalf("hot chat must not be evicted for background: %+v", r)
	}
	bgEmbed.StepDone(ge.Step, nil, Usage{})
	cur.StepDone(g.Step, nil, Usage{})
	_ = h.finish(cur, wait)
	// Idle and past hot_ttl: embed finishes; vision may now evict cold chat.
	h.clock.Advance(300 * time.Millisecond)
	_ = h.runToCompletion(bgEmbed, wait)
	gv := h.grant(bgVision, wait)
	bgVision.StepDone(gv.Step, nil, Usage{})
	_ = h.finish(bgVision, wait)
}

func TestNewRejectsPinnedLargerThanLimit(t *testing.T) {
	cfg, err := config.Parse([]byte(`
gpu: { device: fake, vram_limit_mb: 4000, headroom_mb: 512 }
runtimes:
  sim: { type: fake }
models:
  big:   { runtime: sim, capabilities: [embedding], fake_vram_mb: 3000, pinned: true }
  small: { runtime: sim, capabilities: [embedding], fake_vram_mb: 500, pinned: true }
`))
	if err != nil {
		t.Fatal(err)
	}
	gpu := gpufake.New("FakeGPU", 16000)
	_, err = New(cfg, map[string]runtime.Runtime{"sim": rtfake.New("sim", gpu)}, gpu, nil, nil, Options{})
	if err == nil || !strings.Contains(err.Error(), "pinned models") {
		t.Fatalf("expected pinned-fit error, got %v", err)
	}
	// Same models, no limit: fine (the device decides at runtime).
	cfg2, _ := config.Parse([]byte(`
gpu: { device: fake, headroom_mb: 512 }
runtimes:
  sim: { type: fake }
models:
  big:   { runtime: sim, capabilities: [embedding], fake_vram_mb: 3000, pinned: true }
  small: { runtime: sim, capabilities: [embedding], fake_vram_mb: 500, pinned: true }
`))
	if _, err := New(cfg2, map[string]runtime.Runtime{"sim": rtfake.New("sim", gpu)}, gpu, nil, nil, Options{}); err != nil {
		t.Fatalf("without vram_limit_mb New must succeed: %v", err)
	}
}

// flakyMonitor fails its first n snapshots.
type flakyMonitor struct {
	inner    gpu.Monitor
	failures atomic.Int32
	remain   atomic.Int32
}

func (m *flakyMonitor) Snapshot(ctx context.Context) (gpu.Snapshot, error) {
	if m.remain.Add(-1) >= 0 {
		m.failures.Add(1)
		return gpu.Snapshot{}, errors.New("nvidia-smi: Unable to determine the device handle")
	}
	return m.inner.Snapshot(ctx)
}

func TestStartupRetriesFlakyGPU(t *testing.T) {
	cfg, err := config.Parse([]byte(`
gpu: { device: fake, headroom_mb: 512, poll_interval: 10ms }
runtimes:
  sim: { type: fake, port_range: [46000, 46049] }
models:
  embed: { runtime: sim, capabilities: [embedding], fake_vram_mb: 600, pinned: true }
`))
	if err != nil {
		t.Fatal(err)
	}
	fake := gpufake.New("FakeGPU", 16000)
	mon := &flakyMonitor{inner: fake}
	mon.remain.Store(2) // first two snapshots fail, third succeeds
	rt := rtfake.New("sim", fake)
	s, err := New(cfg, map[string]runtime.Runtime{"sim": rt}, mon, nil, nil, Options{
		Tick: 5 * time.Millisecond, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = s.Run(ctx); close(done) }()
	t.Cleanup(func() { cancel(); <-done })

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		st := s.State()
		for _, r := range st.Resident {
			if r.ID == "embed" && r.State == "ready" {
				if mon.failures.Load() < 2 {
					t.Errorf("expected the monitor to have failed twice, got %d", mon.failures.Load())
				}
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("pinned model never loaded after flaky start; state=%+v", s.State())
}
