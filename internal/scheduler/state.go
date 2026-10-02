package scheduler

import "time"

// State is the JSON document served at /admin/state and rendered by
// `gridcore status`. It is a snapshot built inside the scheduler loop, so it
// is always internally consistent.
type State struct {
	Now      time.Time       `json:"now"`
	Mode     string          `json:"mode"` // interactive | idle
	GPU      GPUState        `json:"gpu"`
	Resident []ResidentModel `json:"resident"`
	Running  []JobState      `json:"running"`
	Queued   []JobState      `json:"queued"`
	Disabled []string        `json:"disabled_models"`
	Events   []Event         `json:"recent_events"`
}

// GPUState summarises the device and the scheduler's view of its memory.
type GPUState struct {
	Name        string    `json:"name"`
	TotalMB     int       `json:"total_mb"`
	BudgetMB    int       `json:"budget_mb"`    // min(total, limit) - headroom
	UsedMB      int       `json:"used_mb"`      // measured, all processes
	CommittedMB int       `json:"committed_mb"` // reservations + resident footprints + external
	AvailableMB int       `json:"available_mb"` // budget - committed
	ExternalMB  int       `json:"external_mb"`  // used by processes GridCore does not manage
	UtilPct     int       `json:"util_pct"`
	SnapshotAt  time.Time `json:"snapshot_at"`
}

// ResidentModel is one loaded (or loading) model.
type ResidentModel struct {
	ID              string    `json:"id"`
	State           string    `json:"state"` // loading | ready | draining | stopping
	Tier            string    `json:"tier"`  // pinned | hot | cold
	VRAMMB          int       `json:"vram_mb"`
	Measured        bool      `json:"measured"` // false while VRAMMB is an estimate/reservation
	Slots           int       `json:"slots"`
	BusySlots       int       `json:"busy_slots"`
	PID             int       `json:"pid,omitempty"`
	Addr            string    `json:"addr,omitempty"`
	LoadedAt        time.Time `json:"loaded_at,omitempty"`
	LoadSeconds     float64   `json:"load_seconds,omitempty"`
	LastUsed        time.Time `json:"last_used,omitempty"`
	LastInteractive time.Time `json:"last_interactive,omitempty"`
	// EvictCost is what evicting the model would cost now (reload seconds x
	// recent demand); the cheapest allowed models are evicted first.
	EvictCost float64 `json:"evict_cost"`
}

// JobState is one queued or running job.
type JobState struct {
	ID         string    `json:"id"`
	Class      string    `json:"class"`
	Kind       string    `json:"kind"`
	Model      string    `json:"model"`
	State      string    `json:"state"`
	Enqueued   time.Time `json:"enqueued"`
	WaitedMS   int64     `json:"waited_ms"` // queue time so far (queued) or until first grant (running)
	Steps      int       `json:"steps"`
	Dispatched int       `json:"dispatched"`
	Completed  int       `json:"completed"`
	MaxWaitMS  int64     `json:"max_wait_ms,omitempty"`
	Reason     string    `json:"waiting_for,omitempty"` // why a queued job is not running yet
}

// Event is one entry of the recent-events ring buffer.
type Event struct {
	At      time.Time `json:"at"`
	Kind    string    `json:"kind"`
	Subject string    `json:"subject"`
	Detail  string    `json:"detail,omitempty"`
}

// Event kinds.
const (
	EvEnqueue   = "enqueue"
	EvDispatch  = "dispatch"
	EvComplete  = "complete"
	EvFail      = "fail"
	EvTimeout   = "timeout"
	EvCancel    = "cancel"
	EvLoad      = "load"
	EvLoaded    = "loaded"
	EvLoadFail  = "load_fail"
	EvEvict     = "evict"
	EvUnloaded  = "unloaded"
	EvCrash     = "crash"
	EvDisabled  = "disabled"
	EvEnabled   = "enabled"
	EvMode      = "mode"
	EvPreempt   = "preempt" // background/batch step deferred because of interactive mode
	EvThrash    = "thrash"  // a model keeps being reloaded: the working set does not fit
	EvShutdown  = "shutdown"
	eventBuffer = 100
)

type eventRing struct {
	buf  []Event
	next int
	full bool
}

func newEventRing(n int) *eventRing { return &eventRing{buf: make([]Event, n)} }

func (r *eventRing) add(e Event) {
	r.buf[r.next] = e
	r.next = (r.next + 1) % len(r.buf)
	if r.next == 0 {
		r.full = true
	}
}

// list returns events oldest first.
func (r *eventRing) list() []Event {
	if !r.full {
		out := make([]Event, r.next)
		copy(out, r.buf[:r.next])
		return out
	}
	out := make([]Event, 0, len(r.buf))
	out = append(out, r.buf[r.next:]...)
	out = append(out, r.buf[:r.next]...)
	return out
}
