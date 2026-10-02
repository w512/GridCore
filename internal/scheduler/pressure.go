package scheduler

import (
	"fmt"
	"time"

	"github.com/w512/gridcore/internal/gpu"
	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/residency"
	"github.com/w512/gridcore/internal/runtime"
)

// Unified memory (Apple Silicon) shares RAM with the OS and every app, so
// the device budget is not the only limit: the host can run short while
// the budget still has room. The monitor reports macOS memory pressure, and
// the scheduler reacts:
//
//	normal   as on a dedicated GPU
//	warn     background and batch work may not load models; it runs on
//	         what is resident (on a 24 GB Mac with apps open, one 12B model
//	         is enough for warn, so this is a common state)
//	critical the same, and one cold model is unloaded every
//	         pressureReliefInterval while it lasts
//
// After such an unload, background and batch loads stay held for
// reliefBackoff even if the reading drops to normal at once: freeing the
// memory is what lowers it, and loading the model straight back raised it
// again (measured on a 24 GB M4 Pro: a 4.8 GB model loaded three times in a
// minute next to a 12B one, swap 5.6 -> 10 GB).
//
// Loads also go one at a time on unified memory, so that the pressure one
// load causes is seen before the next starts (two loads at once took 21 s
// instead of 3 while the system swapped).
//
// Interactive work is never held: a person is waiting and the device
// budget still protects the GPU. The level is raised at once and lowered
// only after pressureCalm of lower readings, so an app that keeps growing
// and shrinking does not make models come and go.
//
// An overcommitted Metal device does not fail loads: running instances
// start failing their requests (all of them, neighbours included). Such an
// out-of-memory report is treated as critical pressure for oomHold, and
// the most likely culprit, a model loaded within oomBlameWindow, is
// unloaded first with its stored measurement discarded.
const (
	pressureCalm           = 10 * time.Second
	pressureReliefInterval = 5 * time.Second
	reliefBackoff          = time.Minute
	oomHold                = 30 * time.Second
	oomBlameWindow         = time.Minute
)

// effectivePressure is the level acted on: the damped reading, or critical
// while a device out-of-memory report is recent.
func (s *Scheduler) effectivePressure(now time.Time) gpu.Pressure {
	if now.Before(s.oomUntil) {
		return gpu.PressureCritical
	}
	return s.pressure
}

// pressureHold says why class c may not load a model now, or "".
func (s *Scheduler) pressureHold(c job.Class, now time.Time) string {
	if c == job.Interactive {
		return ""
	}
	if now.Before(s.reliefUntil) && s.effectivePressure(now) < gpu.PressureWarn {
		return fmt.Sprintf("memory pressure: no new loads for %s for %s after unloading %s", c, s.reliefUntil.Sub(now).Round(time.Second), s.reliefModel)
	}
	if lvl := s.effectivePressure(now); lvl >= gpu.PressureWarn {
		if now.Before(s.oomUntil) {
			return "device out of memory: no new loads for " + string(c)
		}
		return fmt.Sprintf("memory pressure %s: no new loads for %s", lvl, c)
	}
	return ""
}

// updatePressure damps the monitor's reading (see the package comment).
func (s *Scheduler) updatePressure(p gpu.Pressure, now time.Time) {
	switch {
	case p > s.pressure:
		s.pressure = p
		s.pressureLow = time.Time{}
		s.event(EvPressure, "memory", p.String())
		s.log.Info("memory pressure", "level", p.String())
	case p < s.pressure:
		if s.pressureLow.IsZero() {
			s.pressureLow = now
		} else if now.Sub(s.pressureLow) >= pressureCalm {
			s.pressure = p
			s.pressureLow = time.Time{}
			s.event(EvPressure, "memory", p.String())
			s.log.Info("memory pressure", "level", p.String())
		}
	default:
		s.pressureLow = time.Time{}
	}
	s.m.MemoryPressure.Set(float64(s.pressure))
}

// checkOOMs looks for out-of-memory errors that running instances reported
// since the last poll.
func (s *Scheduler) checkOOMs(now time.Time) {
	seen := map[runtime.Instance]bool{}
	var newest *residency.Entry
	total := 0
	for _, e := range s.res.All() {
		rep, ok := e.Instance.(runtime.OOMReporter)
		if !ok || e.State == residency.Stopping {
			continue
		}
		seen[e.Instance] = true
		n := rep.OOMs()
		if d := n - s.ooms[e.Instance]; d > 0 {
			total += d
			s.m.DeviceOOMs.WithLabelValues(e.ID).Add(float64(d))
		}
		s.ooms[e.Instance] = n
		if e.State == residency.Ready && !e.Evicting && (newest == nil || e.LoadedAt.After(newest.LoadedAt)) {
			newest = e
		}
	}
	for inst := range s.ooms {
		if !seen[inst] {
			delete(s.ooms, inst)
		}
	}
	if total == 0 {
		return
	}
	s.oomUntil = now.Add(oomHold)
	s.event(EvOOM, "gpu", fmt.Sprintf("device out of memory (%d errors since last poll); no background loads for %s", total, oomHold))
	s.log.Error("device out of memory: running instances are failing requests", "errors", total)
	if newest != nil && now.Sub(newest.LoadedAt) < oomBlameWindow {
		// The admission size of the latest load was wrong: forget its
		// measurement and unload it before anything else.
		if p, ok := s.store.Get(s.profileKey(newest.Spec)); ok && p.VRAMMB > 0 {
			s.updateProfile(newest.Spec, func(p *model.Profile) { p.VRAMMB = 0 })
		}
		s.evict(newest, "device out of memory after its load")
		s.lastRelief = now
	}
}

// relievePressure unloads one model while the effective level is critical:
// a cold one, or after a device OOM also an idle hot one, cheapest first.
func (s *Scheduler) relievePressure(now time.Time) {
	if s.effectivePressure(now) < gpu.PressureCritical || now.Sub(s.lastRelief) < pressureReliefInterval {
		return
	}
	for _, e := range s.res.All() {
		if e.Evicting || e.State == residency.Loading {
			return // memory is already on its way back; look again next poll
		}
	}
	var victim *residency.Entry
	reason := "memory pressure critical"
	if now.Before(s.oomUntil) {
		// The device is failing requests: anything an interactive request
		// could evict (cold, or hot and idle) may go.
		reason = "device out of memory"
		if vs := s.res.VictimsFor(1, job.Interactive, false, now); len(vs) > 0 {
			victim = vs[0]
		}
	} else {
		victim = s.res.ColdVictim(now)
	}
	if victim == nil {
		return
	}
	s.evict(victim, reason)
	s.lastRelief = now
	s.reliefUntil, s.reliefModel = now.Add(reliefBackoff), victim.ID
}

// loadInFlight reports whether a model is loading. On unified memory loads
// go one at a time (see the package comment).
func (s *Scheduler) loadInFlight() bool {
	for _, e := range s.res.All() {
		if e.State == residency.Loading {
			return true
		}
	}
	return false
}
