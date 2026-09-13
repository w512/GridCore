// Package runtime defines the process-lifecycle contract between the
// scheduler and inference engines.
//
// A Runtime knows how to start a model and hand back an Instance. It knows
// nothing about VRAM (see package gpu) or scheduling. Every practical engine
// (llama-server, vLLM, mlx_lm.server) speaks OpenAI-compatible HTTP, so
// there is no Execute: the API layer reverse-proxies to Instance.Addr().
package runtime

import (
	"context"
	"errors"
	"time"

	"github.com/w512/gridcore/internal/model"
)

// ErrLoadTimeout is returned by Load when the instance did not become
// healthy within the runtime's load timeout.
var ErrLoadTimeout = errors.New("runtime: load timed out")

// Runtime starts model instances.
type Runtime interface {
	// Name is the config key this runtime was created from.
	Name() string
	// Load starts spec on the given local port and blocks until the instance
	// is healthy, ctx is cancelled, or the load timeout elapses. On error no
	// process must be left running.
	Load(ctx context.Context, spec *model.Spec, port int) (Instance, error)
}

// Instance is one running model process.
type Instance interface {
	Spec() *model.Spec
	// Addr is the host:port the proxy should target (127.0.0.1:41003).
	Addr() string
	// PID identifies the process for per-process VRAM attribution.
	PID() int
	// StartedAt is when the process became healthy.
	StartedAt() time.Time
	// Health probes the instance; nil means ready to serve.
	Health(ctx context.Context) error
	// Stop terminates the process: graceful first, then forcibly after a
	// grace period. Callers must have drained running jobs first.
	Stop(ctx context.Context) error
	// Done is closed when the process has exited for any reason.
	Done() <-chan struct{}
	// Err returns the exit error after Done is closed (nil = clean exit).
	Err() error
}

// StopGrace is how long Stop waits after SIGTERM before SIGKILL.
const StopGrace = 5 * time.Second

// Detailed is implemented by load errors that carry extra context (process
// output) which belongs in logs rather than in the one-line error.
type Detailed interface {
	error
	Tail() string
}

// OOMError is implemented by load errors caused by the device running out
// of memory. The scheduler treats it as "the admission size was wrong".
type OOMError interface {
	error
	OOM() bool
}
