package scheduler

import (
	"fmt"
	"time"

	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/residency"
)

// admission is how a variant could be served right now.
type admission int

const (
	admNone      admission = iota // cannot be made resident now
	admEvict                      // would need evictions the class may make
	admBusy                       // resident, all slots busy
	admFree                       // loads into free VRAM
	admLoading                    // already loading
	admReady                      // resident with a free slot
	admHotReady                   // resident with a free slot, in interactive use
	admUnloading                  // resident but being evicted
)

func (a admission) String() string {
	return [...]string{"cannot fit now", "needs evicting", "resident, slots busy", "fits in free VRAM", "loading", "resident", "resident, in interactive use", "unloading"}[a]
}

// admissionOf mirrors ensureLoaded without side effects.
func (s *Scheduler) admissionOf(id string, c job.Class, overdue bool, now time.Time) admission {
	if ent, ok := s.res.Get(id); ok {
		switch {
		case ent.State == residency.Loading:
			return admLoading
		case ent.State != residency.Ready:
			return admUnloading
		case ent.FreeSlots() > 0 && s.res.Tier(ent, now) == residency.Hot:
			return admHotReady
		case ent.FreeSlots() > 0:
			return admReady
		default:
			return admBusy
		}
	}
	if !s.snapOK || s.pressureHold(c, now) != "" {
		return admNone
	}
	sp := s.specs[id]
	need := s.needMB(sp)
	if need > s.maxLoadableMB(sp) {
		return admNone
	}
	avail := s.availableMB()
	if need <= avail {
		return admFree
	}
	if pending := s.pendingFreeMB(); need <= avail+pending {
		return admEvict // memory is already on its way back
	} else if s.res.VictimsFor(need-avail-pending, c, overdue, now) != nil {
		return admEvict
	}
	return admNone
}

// pickVariant commits a family job to one of its variants and reports
// whether it did; otherwise the job keeps waiting and is reconsidered on the
// next pass.
//
// Interactive work gets the best variant that can be had at all, exactly as
// if it had asked for that model: it may evict cold and idle hot models, and
// it waits for a busy or loading best variant rather than settle for less.
// It falls back only when the better variants cannot be made resident (they
// would have to evict pinned or busy hot models, or exceed the budget).
//
// Background and batch work take the least disruptive variant: the best one
// that is resident with a free slot, loading, or fits in free VRAM; then
// one the user is chatting with (its free slot would make the next chat
// request wait); then the best one the class may evict for; then a
// resident one whose slot will free up.
func (s *Scheduler) pickVariant(js *jobState, now time.Time) bool {
	c := js.job.Class
	var usable []string
	adm := map[string]admission{}
	for _, id := range js.job.Variants {
		if _, off := s.disabled[id]; off {
			continue
		}
		usable = append(usable, id)
		adm[id] = s.admissionOf(id, c, s.batchOverdue(js, now), now)
	}
	if len(usable) == 0 {
		s.fail(js, fmt.Errorf("%w: every variant of %s is disabled", ErrModelDisabled, js.job.Family), "disabled")
		return false
	}
	first := func(ok func(admission) bool) string {
		for _, id := range usable {
			if ok(adm[id]) {
				return id
			}
		}
		return ""
	}

	var pick string
	if c == job.Interactive {
		pick = first(func(a admission) bool { return a != admNone })
	} else {
		for _, ok := range []func(admission) bool{
			func(a admission) bool { return a == admReady || a == admLoading || a == admFree },
			func(a admission) bool { return a == admHotReady },
			func(a admission) bool { return a == admEvict },
			func(a admission) bool { return a == admBusy },
		} {
			if pick = first(ok); pick != "" {
				break
			}
		}
	}
	if pick == "" {
		if why := s.pressureHold(c, now); why != "" {
			js.reason = fmt.Sprintf("%s (%s: no variant resident)", why, js.job.Family)
			return false
		}
		js.reason = fmt.Sprintf("waiting for VRAM (%s: no variant fits)", js.job.Family) + s.heldBy(c, now)
		return false
	}

	js.job.ModelID = pick
	detail := fmt.Sprintf("%s -> %s (%s)", js.job.Family, pick, adm[pick])
	if best := usable[0]; best != pick {
		detail += fmt.Sprintf("; %s %s", best, adm[best])
	}
	s.event(EvVariant, js.job.ID, detail)
	s.m.VariantSelected.WithLabelValues(js.job.Family, pick, string(c)).Inc()
	return true
}
