package scheduler

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/residency"
)

const preloadRetryDelay = 5 * time.Second

type preloadState struct {
	nextAttempt time.Time
	status      string // last logged state; waiting must not log every tick
}

// preload records startup intent until each model actually reaches Ready.
func (s *Scheduler) preload() {
	for id, sp := range s.specs {
		if sp.Preload || sp.Pinned {
			s.queuePreload(id)
		}
	}
	s.retryPreloads(s.now())
}

func (s *Scheduler) queuePreload(id string) *preloadState {
	p := s.preloads[id]
	if p == nil {
		p = &preloadState{}
		s.preloads[id] = p
	}
	return p
}

// retryPreloads visits pinned models first. A waiting model does not block
// smaller preloads; unified-memory admission still allows only one load.
func (s *Scheduler) retryPreloads(now time.Time) {
	if s.shuttingDown || len(s.preloads) == 0 {
		return
	}
	ids := sortedSpecIDs(s.specs)
	for _, pinned := range []bool{true, false} {
		for _, id := range ids {
			sp, p := s.specs[id], s.preloads[id]
			if p == nil || sp.Pinned != pinned {
				continue
			}
			if ent, ok := s.res.Get(id); ok {
				switch ent.State {
				case residency.Ready:
					delete(s.preloads, id)
				case residency.Loading:
					s.logPreload(sp, p, "loading", "")
				default:
					s.logPreload(sp, p, "evicting", "waiting for the previous instance to stop")
				}
				continue
			}
			if reason, disabled := s.disabled[id]; disabled {
				s.logPreload(sp, p, "disabled", reason)
				continue
			}
			if now.Before(p.nextAttempt) {
				continue
			}
			if s.interactiveMode(now) {
				s.logPreload(sp, p, "held", "interactive work has priority")
				continue
			}
			if reason := s.loadHold(job.Background, now); reason != "" {
				s.logPreload(sp, p, "held", reason)
				continue
			}
			status, err := s.ensureLoaded(sp, job.Background, false, now)
			if errors.Is(err, ErrModelTooLarge) {
				need, capacity := s.needMB(sp), s.capacityMB(sp)
				if need > capacity {
					// The device/configuration cannot fit it even after all
					// external usage goes away. Do not retry on every tick.
					s.logPreload(sp, p, "failed", fmt.Sprintf("needs %d MB, device/configuration can provide at most %d MB", need, capacity))
					delete(s.preloads, id)
				} else {
					s.logPreload(sp, p, "waiting", "external GPU usage prevents the load")
				}
				continue
			}
			if err != nil {
				p.nextAttempt = now.Add(preloadRetryDelay)
				s.logPreload(sp, p, "retrying", err.Error())
				continue
			}
			s.logPreload(sp, p, status, "")
		}
	}
}

// retryPreloadAfterFailure also restores pinned instances that had already
// reached Ready. Ordinary preloads stop being automatic after their first
// success, so eviction, unload or a later crash cannot cause reload churn.
func (s *Scheduler) retryPreloadAfterFailure(sp *model.Spec, now time.Time) {
	if s.shuttingDown || !sp.Pinned && s.preloads[sp.ID] == nil {
		return
	}
	p := s.queuePreload(sp.ID)
	p.nextAttempt = now.Add(preloadRetryDelay)
	if _, disabled := s.disabled[sp.ID]; !disabled {
		s.logPreload(sp, p, "retrying", fmt.Sprintf("next automatic attempt in %s", preloadRetryDelay))
	}
}

func (s *Scheduler) logPreload(sp *model.Spec, p *preloadState, status, reason string) {
	if p.status == status {
		return
	}
	p.status = status
	level := slog.LevelInfo
	if status == "failed" && sp.Pinned {
		level = slog.LevelError
	} else if status == "failed" || status == "retrying" || status == "disabled" || sp.Pinned && status != "loading" {
		level = slog.LevelWarn
	}
	attrs := []any{"model", sp.ID, "status", status}
	if reason != "" {
		attrs = append(attrs, "reason", reason)
	}
	s.log.Log(context.Background(), level, "preload", attrs...)
}
