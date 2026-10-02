package residency

import (
	"math"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/job"
)

func costSet(opts ...func(*Options)) *Set {
	o := Options{
		HotTTL:       5 * time.Minute,
		MinResidency: 30 * time.Second,
		Cost:         true,
		ClassWeight:  map[job.Class]float64{job.Interactive: 1, job.Background: 0.3, job.Batch: 0.1},
	}
	for _, f := range opts {
		f(&o)
	}
	return NewWithOptions(o)
}

func loadedFor(c job.Class, at time.Time) func(*Entry) {
	return func(e *Entry) { e.LoadedFor = c; e.LoadStarted = at.Add(-2 * time.Second); e.LoadedAt = at }
}

func backgroundAt(t time.Time) func(*Entry) {
	return func(e *Entry) { e.LastBackground = t; e.LastUsed = t }
}

func TestDemandDecays(t *testing.T) {
	s := costSet()
	s.Use("m", job.Interactive, t0)
	if d := s.Demand("m", t0); d != 1 {
		t.Errorf("demand right after one interactive use = %v", d)
	}
	if d := s.Demand("m", t0.Add(reuseHalfLife)); math.Abs(d-0.5) > 1e-9 {
		t.Errorf("demand one half-life later = %v, want 0.5", d)
	}
	s.Use("b", job.Batch, t0)
	if d := s.Demand("b", t0); math.Abs(d-0.1) > 1e-9 {
		t.Errorf("batch use should weigh 0.1, got %v", d)
	}
	if d := s.Demand("never", t0); d != 0 {
		t.Errorf("unknown model demand = %v", d)
	}
}

// The load test thrash: E4B serves background, E2B batch, and they do not
// fit together. Batch must not push out the model background is using.
func TestBatchDoesNotEvictWhatBackgroundUses(t *testing.T) {
	s := costSet()
	now := t0.Add(time.Hour)
	s.Add(entry("e4b", 4200, loadedFor(job.Background, now.Add(-10*time.Minute)), backgroundAt(now.Add(-5*time.Second))))
	if v := s.Victims(1000, job.Batch, now); v != nil {
		t.Errorf("batch evicted a model background used 5s ago: %v", ids(v))
	}
	if v := s.Victims(1000, job.Background, now); !eq(ids(v), []string{"e4b"}) {
		t.Errorf("background may replace it (resident for 10m), got %v", ids(v))
	}
	later := now.Add(31 * time.Second)
	if v := s.Victims(1000, job.Batch, later); !eq(ids(v), []string{"e4b"}) {
		t.Errorf("after min_residency without background use batch may evict it, got %v", ids(v))
	}
}

func TestBackgroundEvictsBatchModelRightAway(t *testing.T) {
	s := costSet()
	now := t0.Add(time.Hour)
	s.Add(entry("e2b", 2900, loadedFor(job.Batch, now.Add(-time.Second))))
	if v := s.Victims(1000, job.Background, now); !eq(ids(v), []string{"e2b"}) {
		t.Errorf("a higher class is not held back by min_residency, got %v", ids(v))
	}
	if v := s.Victims(1000, job.Batch, now); v != nil {
		t.Errorf("same class within min_residency must wait, got %v", ids(v))
	}
}

func TestMinResidencySameClass(t *testing.T) {
	s := costSet()
	now := t0.Add(time.Hour)
	s.Add(entry("a", 5000, loadedFor(job.Background, now.Add(-10*time.Second))))
	if p, why := s.Protected(mustGet(t, s, "a"), job.Background, now); !p || why != "loaded 10s ago" {
		t.Errorf("protected = %v %q", p, why)
	}
	if v := s.Victims(1000, job.Interactive, now); !eq(ids(v), []string{"a"}) {
		t.Errorf("interactive is never held back by min_residency, got %v", ids(v))
	}
	if v := s.Victims(1000, job.Background, now.Add(21*time.Second)); !eq(ids(v), []string{"a"}) {
		t.Errorf("after min_residency it may go, got %v", ids(v))
	}

	off := costSet(func(o *Options) { o.MinResidency = 0 })
	off.Add(entry("a", 5000, loadedFor(job.Background, now.Add(-time.Second))))
	if v := off.Victims(1000, job.Background, now); !eq(ids(v), []string{"a"}) {
		t.Errorf("min_residency 0 disables the rule, got %v", ids(v))
	}
}

// LRU would drop the model used every few seconds because another one was
// touched a moment later; cost keeps the one in demand.
func TestCostKeepsFrequentlyUsedModel(t *testing.T) {
	now := t0.Add(time.Hour)
	build := func(s *Set) {
		s.Add(entry("ambient", 4000, used(now.Add(-3*time.Second))))
		s.Add(entry("once", 4000, used(now.Add(-time.Second))))
		for i := 60; i > 0; i-- {
			s.Use("ambient", job.Background, now.Add(-time.Duration(i)*5*time.Second))
		}
		s.Use("once", job.Background, now.Add(-time.Second))
	}
	lru := New(5 * time.Minute)
	build(lru)
	if v := lru.Victims(4000, job.Interactive, now); !eq(ids(v), []string{"ambient"}) {
		t.Fatalf("precondition: LRU picks the older last use, got %v", ids(v))
	}
	cost := costSet()
	build(cost)
	if v := cost.Victims(4000, job.Interactive, now); !eq(ids(v), []string{"once"}) {
		t.Errorf("cost should keep the model in steady demand, got %v", ids(v))
	}
}

func TestCostWeighsReloadTime(t *testing.T) {
	now := t0.Add(time.Hour)
	s := costSet(func(o *Options) {
		o.ReloadSeconds = func(e *Entry) float64 {
			if e.ID == "big" {
				return 8
			}
			return 1
		}
	})
	s.Add(entry("big", 4000, used(now.Add(-time.Minute))))
	s.Add(entry("small", 4000, used(now.Add(-2*time.Minute))))
	s.Use("big", job.Background, now.Add(-time.Minute))
	s.Use("small", job.Background, now.Add(-time.Minute))
	if v := s.Victims(4000, job.Interactive, now); !eq(ids(v), []string{"small"}) {
		t.Errorf("equal demand: the cheaper reload should go, got %v", ids(v))
	}
}

// Greedy by cost per MB would evict the big model; the small one is enough
// and cheaper.
func TestCostPicksCheapestSufficientSet(t *testing.T) {
	now := t0.Add(time.Hour)
	s := costSet()
	s.Add(entry("big", 8000, used(now.Add(-time.Hour))))
	s.Add(entry("small", 4000, used(now.Add(-time.Hour))))
	for i := 0; i < 6; i++ {
		s.Use("big", job.Background, now.Add(-time.Minute))
	}
	for i := 0; i < 4; i++ {
		s.Use("small", job.Background, now.Add(-time.Minute))
	}
	if v := s.Victims(4000, job.Interactive, now); !eq(ids(v), []string{"small"}) {
		t.Errorf("got %v", ids(v))
	}
	if v := s.Victims(10000, job.Interactive, now); len(v) != 2 {
		t.Errorf("both are needed for 10 GB, got %v", ids(v))
	}
	if v := s.Victims(12001, job.Interactive, now); v != nil {
		t.Errorf("cannot free 12001 MB, got %v", ids(v))
	}
}

func TestCostKeepsTierAndBusyOrder(t *testing.T) {
	now := t0.Add(time.Minute)
	s := costSet()
	s.Add(entry("hot", 5000, interactiveAt(t0)))          // no recorded demand: cost 0
	s.Add(entry("cold-busy", 5000, used(t0), running(1))) // no demand either
	s.Add(entry("cold-idle", 5000, used(t0.Add(-time.Hour))))
	for i := 0; i < 50; i++ {
		s.Use("cold-idle", job.Background, now) // expensive, but still cold and idle
	}
	if v := s.Victims(5000, job.Interactive, now); !eq(ids(v), []string{"cold-idle"}) {
		t.Errorf("cold idle before cold busy before hot, whatever the cost; got %v", ids(v))
	}
	if v := s.Victims(10000, job.Interactive, now); !eq(ids(v), []string{"cold-busy", "cold-idle"}) {
		t.Errorf("then cold busy, got %v", ids(v))
	}
}

func TestCostTiesFallBackToLRU(t *testing.T) {
	now := t0.Add(time.Hour)
	s := costSet()
	s.Add(entry("newer", 4000, used(now.Add(-time.Minute))))
	s.Add(entry("older", 4000, used(now.Add(-2*time.Minute))))
	if v := s.Victims(4000, job.Interactive, now); !eq(ids(v), []string{"older"}) {
		t.Errorf("without demand data the least recently used goes, got %v", ids(v))
	}
}

func mustGet(t *testing.T, s *Set, id string) *Entry {
	t.Helper()
	e, ok := s.Get(id)
	if !ok {
		t.Fatalf("no entry %s", id)
	}
	return e
}
