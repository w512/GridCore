package scheduler

import (
	"strings"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/gpu"
	"github.com/w512/gridcore/internal/job"
)

// Three models that fit together in 16 GB, plus a pinned embedder.
const (
	pmA = "  a: { runtime: sim, capabilities: [chat], simulated_vram_mb: 3000, simulated_load_time: 5ms, parallel: 2 }\n"
	pmB = "  b: { runtime: sim, capabilities: [chat], simulated_vram_mb: 3000, simulated_load_time: 5ms, parallel: 2 }\n"
	pmC = "  c: { runtime: sim, capabilities: [chat], simulated_vram_mb: 3000, simulated_load_time: 5ms, parallel: 2 }\n"
	pmE = "  e: { runtime: sim, capabilities: [embedding], simulated_vram_mb: 500, simulated_load_time: 5ms, parallel: 2, pinned: true }\n"
)

func unifiedHarness(t *testing.T, models string) *harness {
	t.Helper()
	h := newHarness(t, 16000, models)
	h.gpu.SetUnified(true)
	h.eventually(func(st State) bool { return st.GPU.MemoryKind == "unified" && st.GPU.Pressure == "normal" }, "unified device")
	return h
}

func queuedReason(sub string) func(State) bool {
	return func(st State) bool {
		for _, j := range st.Queued {
			if strings.Contains(j.Reason, sub) {
				return true
			}
		}
		return false
	}
}

// Under warn, background work keeps running on resident models but may not
// load new ones; a person waiting may. Back at normal, the hold lasts until
// pressure has been calm for pressureCalm.
func TestPressureWarnHoldsBackgroundLoads(t *testing.T) {
	h := unifiedHarness(t, pmA+pmB+pmC)
	h.run(job.Background, "a")

	h.gpu.SetPressure(gpu.PressureWarn)
	h.eventually(func(st State) bool { return st.GPU.Pressure == "warn" }, "pressure warn")
	h.run(job.Background, "a") // resident: runs as usual

	bg := h.submit(job.Background, "b", 1)
	h.noGrant(bg, 50*time.Millisecond)
	h.eventually(queuedReason("memory pressure warn: no new loads for background"), "background says why it waits")
	h.run(job.Interactive, "c")
	if n := h.loadsOf("b"); n != 0 {
		t.Fatalf("b loaded %d times under warn", n)
	}

	h.gpu.SetPressure(gpu.PressureNormal)
	h.noGrant(bg, 50*time.Millisecond) // lower readings, not yet calm for long
	h.clock.Advance(pressureCalm + time.Second)
	if err := h.runToCompletion(bg, wait); err != nil {
		t.Fatal(err)
	}
	if len(h.events(EvPressure)) < 2 {
		t.Errorf("pressure changes should be events: %v", h.events(EvPressure))
	}
}

// A family request from background work takes a resident variant under
// pressure instead of waiting for a load it may not start.
func TestPressureFamilyPicksResidentVariant(t *testing.T) {
	h := unifiedHarness(t, pmA+pmB)
	h.run(job.Background, "b")
	h.gpu.SetPressure(gpu.PressureWarn)
	h.eventually(func(st State) bool { return st.GPU.Pressure == "warn" }, "pressure warn")

	hd := h.submit(job.Background, "", 1, func(j *job.Job) { j.Family = "fam"; j.Variants = []string{"a", "b"} })
	g := h.grant(hd, wait)
	if g.Model != "b" {
		t.Errorf("got variant %s, want resident b", g.Model)
	}
	hd.StepDone(g.Step, nil, Usage{})
	if n := h.loadsOf("a"); n != 0 {
		t.Errorf("a loaded under warn")
	}
}

// Critical pressure unloads cold models one at a time; hot and pinned ones
// stay. min_residency does not protect a model from the host running out.
func TestPressureCriticalUnloadsColdModels(t *testing.T) {
	h := unifiedHarness(t, pmA+pmB+pmE)
	h.eventually(isReady("e"), "pinned model preloaded")
	h.run(job.Background, "a")  // cold, loaded for background just now
	h.run(job.Interactive, "b") // hot

	h.gpu.SetPressure(gpu.PressureCritical)
	h.eventually(notResident("a"), "the cold model is unloaded")
	if h.resident("b") == nil || h.resident("e") == nil {
		t.Fatalf("hot and pinned models must stay:\n%s", h.dump())
	}

	// b turns cold (hot_ttl 200ms) and the relief interval passes.
	h.clock.Advance(pressureReliefInterval + time.Second)
	h.eventually(notResident("b"), "the next cold model goes after the relief interval")
	if h.resident("e") == nil {
		t.Fatal("pinned model must never be unloaded")
	}
	for _, e := range h.events(EvEvict) {
		if !strings.Contains(e.Detail, "memory pressure critical") {
			t.Errorf("eviction should name the reason: %q", e.Detail)
		}
	}

	// Unloading is what lowers the reading: back at normal, background
	// still may not load for reliefBackoff, or it would raise it again.
	h.gpu.SetPressure(gpu.PressureNormal)
	time.Sleep(50 * time.Millisecond) // polls see the lower reading and start the calm period
	h.clock.Advance(pressureCalm + time.Second)
	h.eventually(func(st State) bool { return st.GPU.Pressure == "normal" }, "pressure normal again")
	bg := h.submit(job.Background, "a", 1)
	h.noGrant(bg, 50*time.Millisecond)
	h.eventually(queuedReason("after unloading b"), "background backs off after the relief")
	h.clock.Advance(reliefBackoff)
	if err := h.runToCompletion(bg, wait); err != nil {
		t.Fatal(err)
	}
}

// The load-test livelock on a 24 GB Mac: an interactive request waits for an
// eviction to finish, a starved background job slips its own load into the
// one load slot, the interactive request evicts that model the moment it is
// ready, and around again: 64 loads in 3 minutes, no chat answered. Lower
// classes must not start loads while interactive work is freeing memory or
// waiting for the load slot.
func TestInteractiveLoadNotStarvedByLowerLoads(t *testing.T) {
	models := "  big: { runtime: sim, capabilities: [chat], simulated_vram_mb: 10000, simulated_load_time: 100ms, parallel: 2 }\n" +
		"  x: { runtime: sim, capabilities: [chat], simulated_vram_mb: 6000, simulated_load_time: 100ms, parallel: 2 }\n" +
		"  y: { runtime: sim, capabilities: [chat], simulated_vram_mb: 5000, simulated_load_time: 100ms, parallel: 2 }\n"
	h := unifiedHarness(t, models)
	bg := h.submit(job.Background, "x", 1)
	h.eventually(func(st State) bool { r := h.resident("x"); return r != nil }, "x loading for background")
	chat := h.submit(job.Interactive, "big", 1)
	batch := h.submit(job.Batch, "y", 1)
	h.clock.Advance(5 * time.Second) // background and batch count as starved from now on
	if err := h.runToCompletion(chat, wait); err != nil {
		t.Fatalf("interactive request: %v\n%s", err, h.dump())
	}
	if n := h.loadsOf("x") + h.loadsOf("y"); n > 2 {
		t.Errorf("lower classes loaded %d times while interactive waited", n)
	}
	h.clock.Advance(time.Second) // big turns cold, interactive mode ends
	for _, hd := range []*Handle{bg, batch} {
		if err := h.runToCompletion(hd, wait); err != nil {
			t.Fatalf("lower classes must still get their turn: %v\n%s", err, h.dump())
		}
	}
}

// On unified memory a second load waits for the first: the pressure one
// load causes must be visible before the next starts.
func TestUnifiedLoadsOneAtATime(t *testing.T) {
	slow := strings.ReplaceAll(pmA+pmB, "simulated_load_time: 5ms", "simulated_load_time: 300ms")
	h := unifiedHarness(t, slow)
	ha := h.submit(job.Interactive, "a", 1)
	hb := h.submit(job.Interactive, "b", 1)
	h.eventually(func(st State) bool {
		loading := 0
		for _, r := range st.Resident {
			if r.State == "loading" {
				loading++
			}
		}
		return loading == 1 && queuedReason("another model to finish loading")(st)
	}, "one load at a time, the other says why it waits")
	for _, hd := range []*Handle{ha, hb} {
		if err := h.runToCompletion(hd, wait); err != nil {
			t.Fatal(err)
		}
	}

	// A dedicated GPU loads both at once.
	d := newHarness(t, 16000, slow)
	da, db := d.submit(job.Interactive, "a", 1), d.submit(job.Interactive, "b", 1)
	d.eventually(func(st State) bool {
		loading := 0
		for _, r := range st.Resident {
			if r.State == "loading" {
				loading++
			}
		}
		return loading == 2
	}, "dedicated GPU loads in parallel")
	for _, hd := range []*Handle{da, db} {
		if err := d.runToCompletion(hd, wait); err != nil {
			t.Fatal(err)
		}
	}
}

// A running instance reports the device out of memory (Metal overcommitted:
// every instance fails its requests). The model loaded just before is
// blamed: unloaded, measurement discarded; background loads wait oomHold.
func TestDeviceOOMUnloadsLatestLoad(t *testing.T) {
	h := unifiedHarness(t, pmA+pmB)
	h.run(job.Interactive, "a")
	h.clock.Advance(2 * time.Minute) // a's load is long past
	h.run(job.Interactive, "b")
	h.eventually(func(st State) bool {
		r := h.resident("b")
		return r != nil && r.Measured
	}, "b measured")

	h.rt.Instance("a").ReportOOM(2)
	h.eventually(notResident("b"), "the latest load is unloaded")
	if h.resident("a") == nil {
		t.Fatalf("the neighbour that reported the OOM stays:\n%s", h.dump())
	}
	if ev := h.events(EvOOM); len(ev) != 1 || !strings.Contains(ev[0].Detail, "2 errors") {
		t.Errorf("oom events: %v", ev)
	}
	if p, ok := h.s.store.Get(h.s.profileKey(h.s.specs["b"])); ok && p.VRAMMB != 0 {
		t.Errorf("b's measurement should be discarded, still %d MB", p.VRAMMB)
	}

	h.clock.Advance(time.Second) // past interactive_idle_before_background, well within oomHold
	bg := h.submit(job.Background, "b", 1)
	h.noGrant(bg, 50*time.Millisecond)
	h.eventually(queuedReason("device out of memory"), "background waits out the OOM hold")
	h.clock.Advance(oomHold + time.Second)
	if err := h.runToCompletion(bg, wait); err != nil {
		t.Fatal(err)
	}
}

// On unified memory other apps' GPU memory swings by hundreds of MB (a
// browser, WindowServer). That must not unload anything or reload anything:
// a model that fits only while the noise is low loads once, and nothing is
// evicted when the noise comes back.
func TestNoisyExternalMemoryCausesNoThrash(t *testing.T) {
	// a and b are pinned so that only the noise could unload them.
	models := "  a: { runtime: sim, capabilities: [chat], simulated_vram_mb: 6000, simulated_load_time: 5ms, parallel: 2, pinned: true }\n" +
		"  b: { runtime: sim, capabilities: [chat], simulated_vram_mb: 6000, simulated_load_time: 5ms, parallel: 2, pinned: true }\n" +
		"  c: { runtime: sim, capabilities: [chat], simulated_vram_mb: 2400, simulated_load_time: 5ms, parallel: 2 }\n"
	h := unifiedHarness(t, models)
	h.eventually(func(st State) bool { return isReady("a")(st) && isReady("b")(st) }, "pinned models preloaded (one at a time)")

	h.gpu.SetExternal(3000) // budget 15488: 12000 + 3000 leaves 488; c (2400 + 512 until measured) waits, does not fail
	time.Sleep(30 * time.Millisecond)
	bg := h.submit(job.Background, "c", 1)
	h.noGrant(bg, 50*time.Millisecond)
	for i := 0; i < 6; i++ { // the browser shrinks and grows again
		h.gpu.SetExternal([]int{500, 3000}[i%2])
		time.Sleep(30 * time.Millisecond)
		h.clock.Advance(time.Second)
	}
	if err := h.runToCompletion(bg, wait); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 6; i++ {
		h.gpu.SetExternal([]int{3000, 500}[i%2])
		time.Sleep(30 * time.Millisecond)
		h.clock.Advance(time.Second)
		h.run(job.Background, "c")
	}
	if n := len(h.events(EvEvict)); n != 0 {
		t.Errorf("noise must not evict anything, got %d evictions:\n%s", n, h.dump())
	}
	for id, want := range map[string]int{"a": 1, "b": 1, "c": 1} {
		if n := h.loadsOf(id); n != want {
			t.Errorf("%s loaded %d times, want %d", id, n, want)
		}
	}
}

// Pressure means nothing on a dedicated GPU, even if a monitor reported it.
func TestPressureIgnoredOnDedicatedGPU(t *testing.T) {
	h := newHarness(t, 16000, pmA+pmB)
	h.gpu.SetPressure(gpu.PressureCritical) // not unified: not reported
	h.run(job.Background, "a")
	h.run(job.Background, "b")
	h.eventually(func(st State) bool { return st.GPU.Pressure == "" && st.GPU.MemoryKind == "dedicated" }, "no pressure on a dedicated device")
}
