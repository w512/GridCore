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
// Implemented in milestone M1.
package residency
