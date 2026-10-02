// Package job defines the scheduler's unit of work and its classification.
//
// Every incoming request is normalised into a Job before the scheduler sees
// it. The API layer builds Jobs; the scheduler owns their lifecycle.
package job

import (
	"context"
	"fmt"
	"time"
)

// Class is the latency/priority class of a job. Classes are strictly ordered:
// all runnable interactive work is considered before any background work,
// and background before batch. There is no cross-class aging in v0.1.
type Class string

const (
	Interactive Class = "interactive"
	Background  Class = "background"
	Batch       Class = "batch"
)

// Classes lists all classes in priority order (highest first).
var Classes = []Class{Interactive, Background, Batch}

// ParseClass validates a user-supplied class string.
func ParseClass(s string) (Class, error) {
	switch Class(s) {
	case Interactive, Background, Batch:
		return Class(s), nil
	}
	return "", fmt.Errorf("unknown class %q (want one of interactive, background, batch)", s)
}

// Preemptible reports whether the scheduler may defer further steps of a job
// of this class once higher-priority work arrives. Interactive work is never
// preempted; background and batch yield between steps.
func (c Class) Preemptible() bool { return c != Interactive }

// Kind is what the job asks the runtime to do. It is derived from the API
// endpoint and used for capability matching against models.
type Kind string

const (
	Chat       Kind = "chat"
	Completion Kind = "completion"
	Embedding  Kind = "embedding"
)

// State is the coarse lifecycle state of a job.
type State string

const (
	Queued    State = "queued"
	Loading   State = "loading" // waiting for the model to become resident
	Running   State = "running"
	Done      State = "done"
	Failed    State = "failed"
	Cancelled State = "cancelled"
	TimedOut  State = "timed_out" // max_wait exceeded while queued
)

// Job is the scheduler's view of a request. It carries no payload: the API
// layer keeps the HTTP request and proxies it once the scheduler grants a
// slot. Jobs with Steps > 1 (chunked embeddings) are dispatched one step at
// a time so higher-priority work can interleave.
type Job struct {
	ID       string
	Class    Class
	Kind     Kind
	ModelID  string        // resolved model id (aliases already applied)
	MaxWait  time.Duration // 0 = unbounded
	Enqueued time.Time

	// Family requests name a model family instead of a model. Variants are
	// the models that may serve it, best first; ModelID stays empty until
	// the scheduler picks one, and is fixed from then on.
	Family   string
	Variants []string

	// Steps is the number of schedulable units (1 for chat/completion,
	// ceil(len(input)/chunk) for embeddings). Completed counts finished steps.
	Steps     int
	Completed int

	// Ctx is cancelled when the client disconnects. The scheduler must drop
	// queued jobs and cancel upstream requests when it fires.
	Ctx context.Context

	State State
}

// New creates a queued job with the given attributes. Steps defaults to 1.
func New(id string, ctx context.Context, class Class, kind Kind, modelID string) *Job {
	return &Job{
		ID:       id,
		Class:    class,
		Kind:     kind,
		ModelID:  modelID,
		Enqueued: time.Now(),
		Steps:    1,
		Ctx:      ctx,
		State:    Queued,
	}
}

// NewFamily creates a queued job for a model family; the scheduler picks
// one of variants (best first).
func NewFamily(id string, ctx context.Context, class Class, kind Kind, family string, variants []string) *Job {
	j := New(id, ctx, class, kind, "")
	j.Family = family
	j.Variants = append([]string(nil), variants...)
	return j
}

// Target is what the job asks for: the model, or the family until a
// variant is picked ("gemma4" / "gemma4:gemma4-e4b").
func (j *Job) Target() string {
	switch {
	case j.Family == "":
		return j.ModelID
	case j.ModelID == "":
		return j.Family
	default:
		return j.Family + ":" + j.ModelID
	}
}

// Deadline returns the wall-clock moment after which a still-queued job must
// be rejected, and ok=false if the job has no max_wait.
func (j *Job) Deadline() (t time.Time, ok bool) {
	if j.MaxWait <= 0 {
		return time.Time{}, false
	}
	return j.Enqueued.Add(j.MaxWait), true
}

// Remaining reports how many steps are still to be dispatched.
func (j *Job) Remaining() int {
	if j.Completed >= j.Steps {
		return 0
	}
	return j.Steps - j.Completed
}

// Terminal reports whether the job has reached a final state.
func (j *Job) Terminal() bool {
	switch j.State {
	case Done, Failed, Cancelled, TimedOut:
		return true
	}
	return false
}
