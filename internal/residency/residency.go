// Package residency tracks which models are loaded and decides who may be
// evicted for whom.
//
// Tiers:
//
//	pinned  configured; preloaded; never evicted
//	hot     served an interactive job within hot_ttl; evictable only for
//	        another interactive job, and only when it has no running jobs
//	cold    everything else; evictable for any class, with two exceptions
//	        that keep models from pushing each other out on every step
//	        (min_residency): batch work does not evict a model background
//	        used that recently, and a model loaded for background (batch)
//	        work is not replaced by other background (batch) work before it
//	        has been resident that long
//
// Among the models the rules allow, victims are the set that is cheapest to
// lose: reload time weighted by recent demand (cost mode), or least
// recently used first (lru mode, the 0.1 behaviour).
//
// The Set is not goroutine-safe: the scheduler loop is its only user.
package residency

import (
	"math"
	"sort"
	"time"

	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/runtime"
)

// State is the lifecycle of a resident entry.
type State string

const (
	Loading  State = "loading"  // Runtime.Load in progress; VRAMMB is a reservation
	Ready    State = "ready"    // serving
	Draining State = "draining" // no new dispatches; waiting for Running to hit 0
	Stopping State = "stopping" // Instance.Stop in progress
)

// Tier is the eviction protection level.
type Tier string

const (
	Pinned Tier = "pinned"
	Hot    Tier = "hot"
	Cold   Tier = "cold"
)

// Entry is one resident (or loading) model.
type Entry struct {
	ID       string
	Spec     *model.Spec
	Instance runtime.Instance // nil while Loading
	State    State
	Port     int

	// VRAMMB is the reservation while loading and the measured (or estimated)
	// footprint afterwards. Always > 0 once admitted.
	VRAMMB int
	// Measured is true once VRAMMB comes from a GPU snapshot rather than an
	// estimate or a stored profile.
	Measured bool

	Slots   int // == Spec.Parallel
	Running int // steps currently dispatched to this instance

	// LoadedFor is the class of the work that caused the load.
	LoadedFor job.Class

	LoadStarted     time.Time
	LoadedAt        time.Time
	LastUsed        time.Time // any dispatch
	LastInteractive time.Time // last interactive dispatch
	LastBackground  time.Time // last background dispatch
	Evicting        bool      // marked as a victim; reason recorded by scheduler
}

// FreeSlots is how many more steps the instance can take right now.
func (e *Entry) FreeSlots() int {
	if e.State != Ready {
		return 0
	}
	if f := e.Slots - e.Running; f > 0 {
		return f
	}
	return 0
}

// Options configure eviction. The zero value (apart from HotTTL) is the
// 0.1 behaviour: LRU order and no minimum residency.
type Options struct {
	HotTTL time.Duration
	// MinResidency bounds model thrash; see the package comment. 0 = off.
	MinResidency time.Duration
	// Cost selects victims by Cost instead of least-recently-used.
	Cost bool
	// ClassWeight scales recent demand per class in Cost. Missing classes
	// weigh 1.
	ClassWeight map[job.Class]float64
	// ReloadSeconds estimates how long reloading e would take (from the
	// stored profile). When nil or <= 0 the entry's own last load time is
	// used.
	ReloadSeconds func(e *Entry) float64
}

// Set is the resident set.
type Set struct {
	opts    Options
	entries map[string]*Entry
	// demand outlives entries: how much a model was wanted matters most
	// right after it was evicted.
	demand map[string]*demand
}

// New creates an empty set with LRU eviction. hotTTL is
// policy.classes.interactive.hot_ttl.
func New(hotTTL time.Duration) *Set { return NewWithOptions(Options{HotTTL: hotTTL}) }

// NewWithOptions creates an empty set.
func NewWithOptions(o Options) *Set {
	return &Set{opts: o, entries: map[string]*Entry{}, demand: map[string]*demand{}}
}

// Add inserts or replaces an entry.
func (s *Set) Add(e *Entry) { s.entries[e.ID] = e }

// Get returns the entry for id, if present.
func (s *Set) Get(id string) (*Entry, bool) {
	e, ok := s.entries[id]
	return e, ok
}

// Remove deletes id.
func (s *Set) Remove(id string) { delete(s.entries, id) }

// Len is the number of entries in any state.
func (s *Set) Len() int { return len(s.entries) }

// All returns entries sorted by id.
func (s *Set) All() []*Entry {
	out := make([]*Entry, 0, len(s.entries))
	for _, e := range s.entries {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// CommittedMB is the sum of every entry's footprint: reservations for
// loading entries and measured/estimated size for the rest. The scheduler
// subtracts this (plus external usage) from the budget to get what is
// available for a new load.
func (s *Set) CommittedMB() int {
	total := 0
	for _, e := range s.entries {
		total += e.VRAMMB
	}
	return total
}

// Tier classifies e at time now.
func (s *Set) Tier(e *Entry, now time.Time) Tier {
	if e.Spec != nil && e.Spec.Pinned {
		return Pinned
	}
	if !e.LastInteractive.IsZero() && now.Sub(e.LastInteractive) < s.opts.HotTTL {
		return Hot
	}
	return Cold
}

// Touch records a dispatch of class c to id.
func (s *Set) Touch(id string, c job.Class, now time.Time) {
	e, ok := s.entries[id]
	if !ok {
		return
	}
	e.LastUsed = now
	switch c {
	case job.Interactive:
		e.LastInteractive = now
	case job.Background:
		e.LastBackground = now
	}
}

// ---- demand ----

// reuseHalfLife is how quickly past requests stop counting as demand. A
// model asked for every few seconds keeps a high score; one asked for once
// ten minutes ago has a quarter of a request left.
const reuseHalfLife = 5 * time.Minute

// demand is a decayed request count per class: each request adds 1 and the
// total halves every reuseHalfLife. It combines how often and how recently
// a model was wanted, which predicts the next request better than recency
// alone (a model used every 5 s is not the one to drop because another was
// touched a second later).
type demand struct {
	n  [3]float64 // indexed like job.Classes
	at time.Time
}

// decayed returns the counts as of now.
func (d *demand) decayed(now time.Time) [3]float64 {
	n := d.n
	if dt := now.Sub(d.at); dt > 0 {
		f := math.Exp2(-dt.Seconds() / reuseHalfLife.Seconds())
		for i := range n {
			n[i] *= f
		}
	}
	return n
}

func classIndex(c job.Class) int {
	for i, k := range job.Classes {
		if k == c {
			return i
		}
	}
	return len(job.Classes) - 1
}

// Use records one request of class c for model id. Call it once per job,
// not per chunk. It is kept after the model is evicted.
func (s *Set) Use(id string, c job.Class, now time.Time) {
	d := s.demand[id]
	if d == nil {
		d = &demand{at: now}
		s.demand[id] = d
	}
	d.n = d.decayed(now)
	d.at = now
	d.n[classIndex(c)]++
}

// Demand is the class-weighted recent request count for id.
func (s *Set) Demand(id string, now time.Time) float64 {
	d := s.demand[id]
	if d == nil {
		return 0
	}
	n := d.decayed(now)
	total := 0.0
	for i, c := range job.Classes {
		w, ok := s.opts.ClassWeight[c]
		if !ok {
			w = 1
		}
		total += w * n[i]
	}
	return total
}

// Cost estimates what evicting e would cost: the seconds it takes to load
// it again, times how much it is in demand (and by whom).
func (s *Set) Cost(e *Entry, now time.Time) float64 {
	return s.reloadSeconds(e) * s.Demand(e.ID, now)
}

func (s *Set) reloadSeconds(e *Entry) float64 {
	if s.opts.ReloadSeconds != nil {
		if r := s.opts.ReloadSeconds(e); r > 0 {
			return r
		}
	}
	if !e.LoadStarted.IsZero() && e.LoadedAt.After(e.LoadStarted) {
		return e.LoadedAt.Sub(e.LoadStarted).Seconds()
	}
	return 1
}

// ---- eviction ----

// Protected reports whether e may not be evicted for work of class c, and
// why. Only Ready entries not already being evicted are ever candidates;
// this covers the policy on top of that.
func (s *Set) Protected(e *Entry, c job.Class, now time.Time) (bool, string) {
	switch s.Tier(e, now) {
	case Pinned:
		return true, "pinned"
	case Hot:
		if c != job.Interactive {
			return true, "hot"
		}
		if e.Running > 0 {
			return true, "hot and busy"
		}
		return false, ""
	}
	if c == job.Interactive || s.opts.MinResidency <= 0 {
		return false, ""
	}
	if c == job.Batch && !e.LastBackground.IsZero() && now.Sub(e.LastBackground) < s.opts.MinResidency {
		return true, "in use by background"
	}
	if e.LoadedFor == c && now.Sub(e.LoadedAt) < s.opts.MinResidency {
		return true, "loaded " + now.Sub(e.LoadedAt).Round(time.Second).String() + " ago"
	}
	return false, ""
}

// maxExactCandidates bounds the exhaustive search in cost mode. A single
// GPU rarely holds more than a handful of models.
const maxExactCandidates = 12

// Victims picks entries to evict so that at least needMB is freed for work
// of class c, or returns nil if the rules cannot free that much. The caller
// marks the returned entries Draining; entries with Running > 0 are stopped
// once they drain.
//
// In cost mode the choice minimises, in this order: hot victims, busy
// victims, total Cost, how recently the newest victim was used (so equal
// costs, e.g. no demand recorded yet, fall back to LRU), and memory freed.
// In lru mode cold idle entries go first, then
// cold busy ones, then hot ones, least recently used first within each.
func (s *Set) Victims(needMB int, c job.Class, now time.Time) []*Entry {
	if needMB <= 0 {
		return []*Entry{}
	}
	var cands []*Entry
	for _, e := range s.entries {
		if e.State != Ready || e.Evicting {
			continue
		}
		if p, _ := s.Protected(e, c, now); p {
			continue
		}
		cands = append(cands, e)
	}
	// Map iteration order is random; make ties deterministic.
	sort.Slice(cands, func(i, j int) bool { return cands[i].ID < cands[j].ID })
	if s.opts.Cost && len(cands) <= maxExactCandidates {
		return s.cheapest(cands, needMB, now)
	}
	return s.lru(cands, needMB, now)
}

func (s *Set) lru(cands []*Entry, needMB int, now time.Time) []*Entry {
	var cold, hot []*Entry
	for _, e := range cands {
		if s.Tier(e, now) == Hot {
			hot = append(hot, e)
		} else {
			cold = append(cold, e)
		}
	}
	// Cold: idle first, then LRU.
	sort.SliceStable(cold, func(i, j int) bool {
		bi, bj := cold[i].Running > 0, cold[j].Running > 0
		if bi != bj {
			return !bi
		}
		return cold[i].LastUsed.Before(cold[j].LastUsed)
	})
	// Hot: LRU by last interactive use.
	sort.SliceStable(hot, func(i, j int) bool {
		return hot[i].LastInteractive.Before(hot[j].LastInteractive)
	})

	var out []*Entry
	freed := 0
	for _, e := range append(cold, hot...) {
		out = append(out, e)
		freed += e.VRAMMB
		if freed >= needMB {
			return out
		}
	}
	return nil
}

// choice scores one candidate set; smaller is better field by field.
type choice struct {
	mask   int
	hot    int
	busy   int
	cost   float64
	freed  int
	newest time.Time
}

func (a choice) less(b choice) bool {
	switch {
	case a.hot != b.hot:
		return a.hot < b.hot
	case a.busy != b.busy:
		return a.busy < b.busy
	case a.cost != b.cost:
		return a.cost < b.cost
	case !a.newest.Equal(b.newest):
		return a.newest.Before(b.newest)
	default:
		return a.freed < b.freed
	}
}

func (s *Set) cheapest(cands []*Entry, needMB int, now time.Time) []*Entry {
	n := len(cands)
	cost := make([]float64, n)
	hot := make([]bool, n)
	for i, e := range cands {
		cost[i] = s.Cost(e, now)
		hot[i] = s.Tier(e, now) == Hot
	}
	best := choice{mask: -1}
	for mask := 1; mask < 1<<n; mask++ {
		ch := choice{mask: mask}
		for i, e := range cands {
			if mask&(1<<i) == 0 {
				continue
			}
			ch.freed += e.VRAMMB
			ch.cost += cost[i]
			if hot[i] {
				ch.hot++
			}
			if e.Running > 0 {
				ch.busy++
			}
			if e.LastUsed.After(ch.newest) {
				ch.newest = e.LastUsed
			}
		}
		if ch.freed < needMB {
			continue
		}
		if best.mask < 0 || ch.less(best) {
			best = ch
		}
	}
	if best.mask < 0 {
		return nil
	}
	var picked []int
	for i := range cands {
		if best.mask&(1<<i) != 0 {
			picked = append(picked, i)
		}
	}
	sort.SliceStable(picked, func(a, b int) bool { return cost[picked[a]] < cost[picked[b]] })
	out := make([]*Entry, len(picked))
	for k, i := range picked {
		out[k] = cands[i]
	}
	return out
}
