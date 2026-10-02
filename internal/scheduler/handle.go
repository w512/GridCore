package scheduler

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/w512/gridcore/internal/job"
)

// Grant tells the API layer that one step of a job may run now on the given
// instance. Steps are numbered 0..Steps-1 in dispatch order.
type Grant struct {
	Step     int
	Model    string // resolved model id
	Addr     string // host:port of the instance
	Instance string // instance key for diagnostics
	Queued   time.Duration
}

// Usage is what the API layer learned from a finished step. Zero values are
// ignored.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	PromptTPS        float64
	GenTPS           float64
}

// Terminal errors reported through Handle.Err.
var (
	ErrQueueTimeout   = errors.New("max_wait exceeded while queued")
	ErrCancelled      = errors.New("cancelled by client")
	ErrModelDisabled  = errors.New("model disabled after repeated failures")
	ErrModelTooLarge  = errors.New("model cannot fit in the VRAM budget")
	ErrLoadFailed     = errors.New("model failed to load")
	ErrInstanceFailed = errors.New("model instance failed")
	ErrShuttingDown   = errors.New("scheduler is shutting down")
	ErrStepFailed     = errors.New("step failed")
)

// Handle is the API layer's side of a submitted job.
//
// Usage pattern:
//
//	h, _ := sched.Submit(j)
//	for {
//	    select {
//	    case g := <-h.Grants():   // run step g.Step against g.Addr, then h.StepDone(...)
//	    case <-h.Done():          // terminal: h.Err() is nil on success
//	    }
//	}
type Handle struct {
	job    *job.Job
	s      *Scheduler
	grants chan Grant
	done   chan struct{}

	mu       sync.Mutex
	err      error
	finished bool
}

func newHandle(s *Scheduler, j *job.Job) *Handle {
	return &Handle{job: j, s: s, grants: make(chan Grant, j.Steps), done: make(chan struct{})}
}

// Job returns the underlying job (read-only after Submit).
func (h *Handle) Job() *job.Job { return h.job }

// Grants delivers one Grant per step. The channel is buffered for all steps
// so the scheduler never blocks on the API layer.
func (h *Handle) Grants() <-chan Grant { return h.grants }

// Done is closed once the job is terminal (all steps completed, or failed,
// timed out, cancelled).
func (h *Handle) Done() <-chan struct{} { return h.done }

// Err is valid after Done is closed. nil means every step completed.
func (h *Handle) Err() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.err
}

// Cancel asks the scheduler to drop the job. Done is closed once the
// scheduler has processed it; grants issued before that must still be
// reported via StepDone (see the API layer's abandon()).
func (h *Handle) Cancel() { h.s.send(evCancel{id: h.job.ID}) }

// StepDone reports the outcome of a granted step. err != nil fails the job;
// remaining in-flight steps are expected to be abandoned by the caller.
func (h *Handle) StepDone(step int, err error, u Usage) {
	h.s.send(evStepDone{id: h.job.ID, step: step, err: err, usage: u})
}

// finish is called by the scheduler loop exactly once.
func (h *Handle) finish(err error) {
	h.mu.Lock()
	if h.finished {
		h.mu.Unlock()
		return
	}
	h.finished = true
	h.err = err
	h.mu.Unlock()
	close(h.done)
}

// jobState is the scheduler's private bookkeeping for a job.
type jobState struct {
	job        *job.Job
	handle     *Handle
	dispatched int
	completed  int
	inflight   int
	failed     bool
	failErr    error
	firstGrant time.Time
	// lastProgress is when the job was enqueued or last finished a step;
	// the starvation guard measures from here.
	lastProgress time.Time
	reason       string // why it is waiting (for /admin/state)

	// instances that hold steps of this job, for slot return on completion
	steps     map[int]string    // step -> resident entry id
	stepStart map[int]time.Time // step -> dispatch time
}

func (js *jobState) remaining() int { return js.job.Steps - js.dispatched }

func (js *jobState) String() string {
	return fmt.Sprintf("%s(%s %s %s %d/%d)", js.job.ID, js.job.Class, js.job.Kind, js.job.Target(), js.completed, js.job.Steps)
}
