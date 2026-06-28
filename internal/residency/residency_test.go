package residency

import (
	"testing"
	"time"

	"github.com/gridcore/gridcore/internal/job"
	"github.com/gridcore/gridcore/internal/model"
)

var t0 = time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)

func entry(id string, vram int, opts ...func(*Entry)) *Entry {
	e := &Entry{ID: id, Spec: &model.Spec{ID: id, Parallel: 2}, State: Ready, VRAMMB: vram, Slots: 2, LastUsed: t0}
	for _, o := range opts {
		o(e)
	}
	return e
}

func pinned(e *Entry)               { e.Spec.Pinned = true }
func running(n int) func(*Entry)    { return func(e *Entry) { e.Running = n } }
func state(s State) func(*Entry)    { return func(e *Entry) { e.State = s } }
func used(t time.Time) func(*Entry) { return func(e *Entry) { e.LastUsed = t } }
func interactiveAt(t time.Time) func(*Entry) {
	return func(e *Entry) { e.LastInteractive = t; e.LastUsed = t }
}

func ids(es []*Entry) []string {
	out := make([]string, len(es))
	for i, e := range es {
		out[i] = e.ID
	}
	return out
}

func eq(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestTier(t *testing.T) {
	s := New(5 * time.Minute)
	now := t0.Add(10 * time.Minute)
	cases := []struct {
		e    *Entry
		want Tier
	}{
		{entry("p", 1, pinned), Pinned},
		{entry("p2", 1, pinned, interactiveAt(now)), Pinned},
		{entry("h", 1, interactiveAt(now.Add(-time.Minute))), Hot},
		{entry("edge", 1, interactiveAt(now.Add(-5*time.Minute))), Cold}, // exactly hot_ttl ago -> cold
		{entry("c", 1, interactiveAt(now.Add(-6*time.Minute))), Cold},
		{entry("never", 1), Cold},
	}
	for _, c := range cases {
		if got := s.Tier(c.e, now); got != c.want {
			t.Errorf("%s: tier = %s want %s", c.e.ID, got, c.want)
		}
	}
}

func TestTouch(t *testing.T) {
	s := New(time.Minute)
	e := entry("m", 1)
	s.Add(e)
	s.Touch("m", job.Background, t0.Add(time.Second))
	if !e.LastUsed.Equal(t0.Add(time.Second)) || !e.LastInteractive.IsZero() {
		t.Errorf("background touch: %+v", e)
	}
	s.Touch("m", job.Interactive, t0.Add(2*time.Second))
	if !e.LastInteractive.Equal(t0.Add(2 * time.Second)) {
		t.Errorf("interactive touch: %+v", e)
	}
	s.Touch("missing", job.Interactive, t0) // no panic
}

func TestVictimsPinnedNever(t *testing.T) {
	s := New(time.Minute)
	s.Add(entry("pin", 8000, pinned))
	if v := s.Victims(100, job.Interactive, t0); v != nil {
		t.Errorf("pinned must never be a victim, got %v", ids(v))
	}
}

func TestVictimsHotProtectedFromBackground(t *testing.T) {
	s := New(5 * time.Minute)
	now := t0.Add(time.Minute)
	s.Add(entry("hot", 9000, interactiveAt(t0)))
	if v := s.Victims(100, job.Background, now); v != nil {
		t.Errorf("hot must not be evicted for background, got %v", ids(v))
	}
	if v := s.Victims(100, job.Batch, now); v != nil {
		t.Errorf("hot must not be evicted for batch, got %v", ids(v))
	}
	v := s.Victims(100, job.Interactive, now)
	if !eq(ids(v), []string{"hot"}) {
		t.Errorf("hot may be evicted for interactive, got %v", ids(v))
	}
}

func TestVictimsHotWithRunningNeverEvenForInteractive(t *testing.T) {
	s := New(5 * time.Minute)
	now := t0.Add(time.Minute)
	s.Add(entry("hot-busy", 9000, interactiveAt(t0), running(1)))
	if v := s.Victims(100, job.Interactive, now); v != nil {
		t.Errorf("hot with running steps must not be a victim, got %v", ids(v))
	}
}

func TestVictimsColdLRUIdleFirst(t *testing.T) {
	s := New(time.Minute)
	now := t0.Add(time.Hour)
	s.Add(entry("old-busy", 3000, used(t0), running(1)))
	s.Add(entry("newer-idle", 3000, used(t0.Add(10*time.Minute))))
	s.Add(entry("oldest-idle", 3000, used(t0.Add(-10*time.Minute))))
	s.Add(entry("newest-idle", 3000, used(t0.Add(20*time.Minute))))

	v := s.Victims(3000, job.Background, now)
	if !eq(ids(v), []string{"oldest-idle"}) {
		t.Errorf("single victim should be LRU idle, got %v", ids(v))
	}
	v = s.Victims(9000, job.Background, now)
	if !eq(ids(v), []string{"oldest-idle", "newer-idle", "newest-idle"}) {
		t.Errorf("idle before busy, LRU within, got %v", ids(v))
	}
	v = s.Victims(10000, job.Background, now)
	if !eq(ids(v), []string{"oldest-idle", "newer-idle", "newest-idle", "old-busy"}) {
		t.Errorf("busy cold last, got %v", ids(v))
	}
}

func TestVictimsColdBeforeHotForInteractive(t *testing.T) {
	s := New(5 * time.Minute)
	now := t0.Add(time.Minute)
	s.Add(entry("hot-old", 5000, interactiveAt(t0.Add(-time.Minute))))
	s.Add(entry("hot-new", 5000, interactiveAt(t0)))
	s.Add(entry("cold", 5000, used(t0.Add(-time.Hour))))

	v := s.Victims(5000, job.Interactive, now)
	if !eq(ids(v), []string{"cold"}) {
		t.Errorf("cold first, got %v", ids(v))
	}
	v = s.Victims(10000, job.Interactive, now)
	if !eq(ids(v), []string{"cold", "hot-old"}) {
		t.Errorf("then LRU hot, got %v", ids(v))
	}
}

func TestVictimsInsufficient(t *testing.T) {
	s := New(time.Minute)
	s.Add(entry("a", 1000))
	s.Add(entry("b", 1000))
	if v := s.Victims(2001, job.Batch, t0); v != nil {
		t.Errorf("should return nil when rules cannot free enough, got %v", ids(v))
	}
	if v := s.Victims(2000, job.Batch, t0); len(v) != 2 {
		t.Errorf("exact fit should work, got %v", ids(v))
	}
}

func TestVictimsSkipNonReadyAndEvicting(t *testing.T) {
	s := New(time.Minute)
	s.Add(entry("loading", 5000, state(Loading)))
	s.Add(entry("draining", 5000, state(Draining)))
	s.Add(entry("stopping", 5000, state(Stopping)))
	ev := entry("evicting", 5000)
	ev.Evicting = true
	s.Add(ev)
	if v := s.Victims(1, job.Interactive, t0); v != nil {
		t.Errorf("non-ready/evicting entries are not candidates, got %v", ids(v))
	}
}

func TestVictimsZeroNeed(t *testing.T) {
	s := New(time.Minute)
	s.Add(entry("a", 1000))
	if v := s.Victims(0, job.Batch, t0); v == nil || len(v) != 0 {
		t.Errorf("zero need should return empty non-nil slice, got %v", v)
	}
}

func TestCommittedAndFreeSlots(t *testing.T) {
	s := New(time.Minute)
	s.Add(entry("a", 1000, running(2)))
	s.Add(entry("b", 2000, state(Loading)))
	if s.CommittedMB() != 3000 {
		t.Errorf("committed = %d", s.CommittedMB())
	}
	a, _ := s.Get("a")
	b, _ := s.Get("b")
	if a.FreeSlots() != 0 || b.FreeSlots() != 0 {
		t.Errorf("free slots a=%d b=%d", a.FreeSlots(), b.FreeSlots())
	}
	a.Running = 1
	if a.FreeSlots() != 1 {
		t.Errorf("free slots a=%d", a.FreeSlots())
	}
	if len(s.All()) != 2 || s.All()[0].ID != "a" || s.Len() != 2 {
		t.Errorf("All/Len mismatch")
	}
	s.Remove("a")
	if _, ok := s.Get("a"); ok {
		t.Error("remove failed")
	}
}
