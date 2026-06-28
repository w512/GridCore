// Package residency tracks which models are loaded and decides who may be
// evicted for whom.
//
// Tiers:
//
//	pinned  configured; preloaded; never evicted
//	hot     served an interactive job within hot_ttl; evictable only for
//	        another interactive job, and only when it has no running jobs
//	cold    everything else; evictable for any class, LRU first
//
// v0.1 orders victims by last_used within a tier. The cost-based score
// (reload_time x p(reuse) x priority) is deferred to v0.2.
//
// The Set is not goroutine-safe: the scheduler loop is its only user.
package residency

import (
	"sort"
	"time"

	"github.com/gridcore/gridcore/internal/job"
	"github.com/gridcore/gridcore/internal/model"
	"github.com/gridcore/gridcore/internal/runtime"
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

	LoadStarted     time.Time
	LoadedAt        time.Time
	LastUsed        time.Time // any dispatch
	LastInteractive time.Time // last interactive dispatch
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

// Set is the resident set.
type Set struct {
	hotTTL  time.Duration
	entries map[string]*Entry
}

// New creates an empty set. hotTTL is policy.classes.interactive.hot_ttl.
func New(hotTTL time.Duration) *Set {
	return &Set{hotTTL: hotTTL, entries: map[string]*Entry{}}
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
	if !e.LastInteractive.IsZero() && now.Sub(e.LastInteractive) < s.hotTTL {
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
	if c == job.Interactive {
		e.LastInteractive = now
	}
}

// Victims picks entries to evict so that at least needMB is freed for work
// of class c. Rules:
//
//   - only Ready entries not already Evicting are candidates
//   - Pinned never
//   - Hot only when c is Interactive and the entry has no running steps
//   - Cold always; idle ones (Running == 0) before busy ones, LRU within each
//
// Returns nil if the rules cannot free needMB. The caller marks the returned
// entries Draining; entries with Running > 0 are stopped once they drain.
func (s *Set) Victims(needMB int, c job.Class, now time.Time) []*Entry {
	if needMB <= 0 {
		return []*Entry{}
	}
	var cold, hot []*Entry
	for _, e := range s.entries {
		if e.State != Ready || e.Evicting {
			continue
		}
		switch s.Tier(e, now) {
		case Pinned:
			continue
		case Hot:
			if c == job.Interactive && e.Running == 0 {
				hot = append(hot, e)
			}
		case Cold:
			cold = append(cold, e)
		}
	}
	// Cold: idle first, then LRU.
	sort.Slice(cold, func(i, j int) bool {
		bi, bj := cold[i].Running > 0, cold[j].Running > 0
		if bi != bj {
			return !bi
		}
		return cold[i].LastUsed.Before(cold[j].LastUsed)
	})
	// Hot: LRU by last interactive use.
	sort.Slice(hot, func(i, j int) bool {
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
