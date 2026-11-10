package scheduler

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/config"
	"github.com/w512/gridcore/internal/gpu"
	gpusim "github.com/w512/gridcore/internal/gpu/simulation"
	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/runtime"
	rtsim "github.com/w512/gridcore/internal/runtime/simulation"
)

const preloadEmbed = "  embed: {runtime: sim, capabilities: [embedding], simulated_vram_mb: 600, simulated_load_time: 5ms, preload: true}\n"

func TestPreloadRetriesAfterExternalMemoryIsFreed(t *testing.T) {
	for _, models := range []string{pinned, preloadEmbed} {
		for _, unified := range []bool{false, true} {
			t.Run(fmt.Sprintf("pinned=%v/unified=%v", models == pinned, unified), func(t *testing.T) {
				h := newHarnessWith(t, 16000, models, "", func(mon *gpusim.Monitor, _ *rtsim.Runtime) {
					mon.SetUnified(unified)
					mon.SetExternal(15000)
				})
				h.eventually(func(st State) bool { return st.GPU.ExternalMB == 15000 }, "GPU initially occupied")
				if h.resident("embed") != nil {
					t.Fatal("preload started without enough free memory")
				}
				h.gpu.SetExternal(0)
				h.eventually(isReady("embed"), "preload recovers without a request")
				if n := h.loadsOf("embed"); n != 1 {
					t.Fatalf("preload started %d loads, want one", n)
				}
			})
		}
	}
}

func TestWaitingPreloadDoesNotBlockSmallerModels(t *testing.T) {
	models := "  a: {runtime: sim, capabilities: [chat], simulated_vram_mb: 3000, preload: true}\n" +
		"  b: {runtime: sim, capabilities: [chat], simulated_vram_mb: 500, preload: true}\n"
	h := newHarnessWith(t, 16000, models, "", func(mon *gpusim.Monitor, _ *rtsim.Runtime) {
		mon.SetExternal(14000)
	})
	h.eventually(isReady("b"), "small preload can run while a larger one waits")
	if h.resident("a") != nil {
		t.Fatal("larger preload should still be waiting")
	}
	h.gpu.SetExternal(0)
	h.eventually(isReady("a"), "larger preload resumes when memory is freed")
}

func TestUnifiedPreloadSurvivesEvictionWait(t *testing.T) {
	models := "  a: {runtime: sim, capabilities: [chat], simulated_vram_mb: 4000, preload: true}\n" +
		"  b: {runtime: sim, capabilities: [chat], simulated_vram_mb: 4000, preload: true}\n"
	h := newHarnessWith(t, 7000, models, "  min_residency: 0s\n", func(mon *gpusim.Monitor, _ *rtsim.Runtime) {
		mon.SetUnified(true)
	})
	h.eventually(isReady("b"), "second preload completes after evicting the first")
	if h.resident("a") != nil || h.loadsOf("a") != 1 || h.loadsOf("b") != 1 {
		t.Fatalf("preloads were lost or repeatedly reloaded:%s", h.dump())
	}
}

func TestPreloadLoadFailureBacksOffUntilSuccess(t *testing.T) {
	h := newHarnessWith(t, 16000, preloadEmbed, "", func(_ *gpusim.Monitor, rt *rtsim.Runtime) {
		rt.SetFailLoad(errors.New("transient load failure"))
	})
	h.eventually(func(st State) bool { return len(h.events(EvLoadFail)) == 1 }, "first load fails")
	h.rt.SetFailLoad(nil)
	h.clock.Advance(preloadRetryDelay - time.Millisecond)
	time.Sleep(30 * time.Millisecond)
	if h.resident("embed") != nil || h.loadsOf("embed") != 1 {
		t.Fatal("preload retried before the backoff expired")
	}
	h.clock.Advance(time.Millisecond)
	h.eventually(isReady("embed"), "failed preload retries without a request")
	if h.loadsOf("embed") != 2 {
		t.Fatalf("want two load attempts:%s", h.dump())
	}
}

func TestPinnedRecoversAfterCrashWithoutRequest(t *testing.T) {
	h := newHarness(t, 16000, pinned)
	h.eventually(isReady("embed"), "initial pinned preload")
	first := h.rt.Instance("embed")
	first.Kill()
	h.eventually(func(st State) bool { return len(h.events(EvCrash)) == 1 }, "pinned crash processed")
	if h.resident("embed") != nil {
		t.Fatal("pinned model retried immediately after its crash")
	}
	h.clock.Advance(preloadRetryDelay)
	h.eventually(func(st State) bool {
		next := h.rt.Instance("embed")
		return next != nil && next != first && isReady("embed")(st)
	}, "pinned model restored without a request")
}

func TestPinnedRecoveryWaitsAfterEmergencyUnloadAndStopsAtShutdown(t *testing.T) {
	h := newHarnessWith(t, 16000, pinned, "", func(mon *gpusim.Monitor, _ *rtsim.Runtime) {
		mon.SetUnified(true)
	})
	h.eventually(isReady("embed"), "initial pinned preload")
	h.rt.Instance("embed").ReportOOM(1)
	h.eventually(notResident("embed"), "emergency OOM unload")
	h.clock.Advance(preloadRetryDelay)
	time.Sleep(30 * time.Millisecond)
	if h.resident("embed") != nil || h.loadsOf("embed") != 1 {
		t.Fatal("pinned recovery bypassed the OOM hold")
	}
	h.clock.Advance(oomHold + time.Second)
	h.eventually(isReady("embed"), "pinned model restored once the OOM hold expires")
	h.cancel()
	select {
	case <-h.done:
	case <-time.After(wait):
		t.Fatal("scheduler did not shut down")
	}
	if len(h.s.preloads) != 0 || h.s.ports.Len() != 0 {
		t.Fatal("shutdown queued pinned recovery or left reserved ports")
	}
}

func TestPreloadRetriesRespectCircuitBreakerAndEnable(t *testing.T) {
	for _, models := range []string{pinned, preloadEmbed} {
		t.Run(fmt.Sprintf("pinned=%v", models == pinned), func(t *testing.T) {
			h := newHarnessWith(t, 16000, models, "", func(_ *gpusim.Monitor, rt *rtsim.Runtime) {
				rt.SetFailLoad(errors.New("broken model"))
			})
			for attempt := 1; attempt <= 3; attempt++ {
				if attempt > 1 {
					h.clock.Advance(preloadRetryDelay)
				}
				h.eventually(func(st State) bool { return len(h.events(EvLoadFail)) == attempt }, "automatic load failure")
			}
			h.eventually(func(st State) bool { return len(st.Disabled) == 1 }, "circuit breaker disables the model")
			h.clock.Advance(10 * time.Minute)
			time.Sleep(30 * time.Millisecond)
			if h.loadsOf("embed") != 3 {
				t.Fatal("automatic preload bypassed the circuit breaker")
			}
			h.rt.SetFailLoad(nil)
			if err := h.s.EnableModel("embed"); err != nil {
				t.Fatal(err)
			}
			h.eventually(isReady("embed"), "enable resumes the pending preload without a request")
		})
	}
}

func TestPreloadIsOneShotAfterReady(t *testing.T) {
	for _, crash := range []bool{false, true} {
		t.Run(fmt.Sprintf("crash=%v", crash), func(t *testing.T) {
			h := newHarness(t, 16000, preloadEmbed)
			h.eventually(isReady("embed"), "initial preload")
			if crash {
				h.rt.Instance("embed").Kill()
			} else if err := h.s.UnloadModel("embed"); err != nil {
				t.Fatal(err)
			}
			h.eventually(notResident("embed"), "preloaded model stopped")
			h.clock.Advance(time.Minute)
			time.Sleep(30 * time.Millisecond)
			if h.resident("embed") != nil || h.loadsOf("embed") != 1 {
				t.Fatal("ordinary preload was restored after its first successful load")
			}
		})
	}
}

func TestPendingPreloadRespectsMemoryPressure(t *testing.T) {
	h := newHarnessWith(t, 16000, pinned, "", func(mon *gpusim.Monitor, _ *rtsim.Runtime) {
		mon.SetUnified(true)
		mon.SetPressure(gpu.PressureWarn)
	})
	h.eventually(func(st State) bool { return st.GPU.Pressure == "warn" }, "startup memory pressure")
	if h.loadsOf("embed") != 0 {
		t.Fatal("preload ignored memory pressure")
	}
	h.gpu.SetPressure(gpu.PressureNormal)
	changed := time.Now()
	h.eventually(func(st State) bool { return st.GPU.SnapshotAt.After(changed) }, "lower pressure snapshot starts the calm interval")
	h.clock.Advance(pressureCalm + time.Second)
	h.eventually(isReady("embed"), "preload resumes once memory pressure calms")
}

func TestPendingPreloadYieldsToInteractiveWork(t *testing.T) {
	h := newHarnessWith(t, 16000, chat+preloadEmbed, "", func(mon *gpusim.Monitor, _ *rtsim.Runtime) {
		mon.SetUnified(true)
		mon.SetExternal(15000)
	})
	h.eventually(func(st State) bool { return st.GPU.ExternalMB == 15000 }, "GPU initially occupied")
	hd := h.submit(job.Interactive, "chat", 1)
	h.eventually(queuedReason("waiting for VRAM"), "interactive request waiting for memory")
	h.gpu.SetExternal(0)
	g := h.grant(hd, wait)
	if h.loadsOf("embed") != 0 {
		t.Fatal("preload took the load slot before interactive work completed")
	}
	hd.StepDone(g.Step, nil, Usage{})
	if err := h.finish(hd, wait); err != nil {
		t.Fatal(err)
	}
	h.clock.Advance(h.cfg.Policy.InteractiveIdleBeforeBackground + time.Second)
	h.eventually(isReady("embed"), "preload resumes after interactive work")
}

func TestPreloadWaitingAndPermanentFailureLogOnce(t *testing.T) {
	for _, tooLarge := range []bool{false, true} {
		t.Run(fmt.Sprintf("too_large=%v", tooLarge), func(t *testing.T) {
			models := pinned
			if tooLarge {
				models = strings.ReplaceAll(pinned, "600", "20000")
			}
			lo := int(testPortBase.Add(20))
			cfg, err := config.Parse([]byte(fmt.Sprintf(baseYAML, lo, lo+19, models)))
			if err != nil {
				t.Fatal(err)
			}
			mon := gpusim.New("Simulated GPU", 16000)
			var logs bytes.Buffer
			s, err := New(cfg, map[string]runtime.Runtime{"sim": rtsim.New("sim", mon)}, mon, nil, nil, Options{
				Logger: slog.New(slog.NewTextHandler(&logs, nil)),
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(s.cancelLoads)
			if tooLarge {
				if err := s.snapshotSync(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			s.preload()
			for range 10 {
				s.retryPreloads(s.now())
			}
			status, level, pending := "waiting", "WARN", 1
			if tooLarge {
				status, level, pending = "failed", "ERROR", 0
			}
			if strings.Count(logs.String(), "status="+status) != 1 || !strings.Contains(logs.String(), "level="+level) || len(s.preloads) != pending {
				t.Fatalf("pending=%d logs=%s", len(s.preloads), logs.String())
			}
		})
	}
}
