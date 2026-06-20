// Package fake is a simulated inference runtime.
//
// It behaves like llama-server from the scheduler's point of view: Load takes
// the model's fake_load_time, the instance serves a minimal OpenAI-compatible
// HTTP API on the assigned port, and its "VRAM" (fake_vram_mb) is registered
// with a fake GPU monitor by PID. This lets the whole scheduler run and be
// tested on a machine with no GPU at all.
package fake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	gpufake "github.com/gridcore/gridcore/internal/gpu/fake"
	"github.com/gridcore/gridcore/internal/model"
	"github.com/gridcore/gridcore/internal/runtime"
)

var nextPID atomic.Int64

func init() { nextPID.Store(100000) }

// Runtime implements runtime.Runtime.
type Runtime struct {
	name string
	gpu  *gpufake.Monitor

	// RequestDelay is how long each simulated request takes. Tests use small
	// values; demos use realistic ones.
	RequestDelay time.Duration
	// LoadTimeout bounds Load regardless of fake_load_time.
	LoadTimeout time.Duration
	// FailLoad, if set, makes every Load fail after the load delay. Tests use
	// it to exercise crash/retry paths.
	FailLoad error
}

// New creates a fake runtime that reports VRAM into gpu.
func New(name string, gpu *gpufake.Monitor) *Runtime {
	return &Runtime{name: name, gpu: gpu, RequestDelay: 20 * time.Millisecond, LoadTimeout: 30 * time.Second}
}

func (r *Runtime) Name() string { return r.name }

// Load implements runtime.Runtime.
func (r *Runtime) Load(ctx context.Context, spec *model.Spec, port int) (runtime.Instance, error) {
	ctx, cancel := context.WithTimeout(ctx, r.LoadTimeout)
	defer cancel()

	// Simulate weights loading.
	select {
	case <-time.After(spec.FakeLoad):
	case <-ctx.Done():
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil, runtime.ErrLoadTimeout
		}
		return nil, ctx.Err()
	}
	if r.FailLoad != nil {
		return nil, r.FailLoad
	}

	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return nil, fmt.Errorf("fake runtime: listen: %w", err)
	}

	inst := &Instance{
		spec:    spec,
		addr:    ln.Addr().String(),
		pid:     int(nextPID.Add(1)),
		started: time.Now(),
		done:    make(chan struct{}),
		delay:   r.RequestDelay,
		gpu:     r.gpu,
	}
	inst.srv = &http.Server{Handler: inst.handler()}
	go func() {
		err := inst.srv.Serve(ln)
		if !errors.Is(err, http.ErrServerClosed) {
			inst.setErr(err)
		}
		inst.closeDone()
	}()
	if r.gpu != nil {
		r.gpu.Attach(inst.pid, spec.FakeVRAMMB)
	}
	return inst, nil
}

// Instance implements runtime.Instance.
type Instance struct {
	spec    *model.Spec
	addr    string
	pid     int
	started time.Time
	delay   time.Duration
	gpu     *gpufake.Monitor
	srv     *http.Server

	inflight atomic.Int32
	peak     atomic.Int32
	requests atomic.Int64

	doneOnce sync.Once
	done     chan struct{}
	errMu    sync.Mutex
	err      error
}

func (i *Instance) Spec() *model.Spec     { return i.spec }
func (i *Instance) Addr() string          { return i.addr }
func (i *Instance) PID() int              { return i.pid }
func (i *Instance) StartedAt() time.Time  { return i.started }
func (i *Instance) Done() <-chan struct{} { return i.done }

func (i *Instance) Err() error {
	i.errMu.Lock()
	defer i.errMu.Unlock()
	return i.err
}

// Health implements runtime.Instance.
func (i *Instance) Health(ctx context.Context) error {
	select {
	case <-i.done:
		return errors.New("fake instance: exited")
	default:
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+i.addr+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("fake instance: health %d", resp.StatusCode)
	}
	return nil
}

// Stop implements runtime.Instance.
//
// It waits for in-flight requests (up to runtime.StopGrace) and then closes
// the server outright. We deliberately avoid http.Server.Shutdown: Go's
// Transport can leave a freshly dialled, never-used keep-alive connection
// behind, which Shutdown treats as "new" and waits on for 5 s (Go issue
// 22682). A real process gets SIGTERM and does not have that problem.
func (i *Instance) Stop(ctx context.Context) error {
	if i.gpu != nil {
		i.gpu.Detach(i.pid)
	}
	i.srv.SetKeepAlivesEnabled(false)

	deadline := time.NewTimer(runtime.StopGrace)
	defer deadline.Stop()
	tick := time.NewTicker(2 * time.Millisecond)
	defer tick.Stop()
	var err error
wait:
	for i.inflight.Load() > 0 {
		select {
		case <-tick.C:
		case <-deadline.C:
			err = errors.New("fake instance: stop grace exceeded with requests in flight")
			break wait
		case <-ctx.Done():
			err = ctx.Err()
			break wait
		}
	}
	_ = i.srv.Close()
	<-i.done
	return err
}

// Kill simulates a crash: the process disappears without Stop being called.
func (i *Instance) Kill() {
	if i.gpu != nil {
		i.gpu.Detach(i.pid)
	}
	i.setErr(errors.New("fake instance: killed"))
	_ = i.srv.Close()
}

// PeakConcurrency reports the maximum number of simultaneous requests seen.
// Tests use it to assert the scheduler respected `parallel`.
func (i *Instance) PeakConcurrency() int { return int(i.peak.Load()) }

// Requests reports the total number of inference requests served.
func (i *Instance) Requests() int { return int(i.requests.Load()) }

func (i *Instance) setErr(err error) {
	i.errMu.Lock()
	defer i.errMu.Unlock()
	if i.err == nil {
		i.err = err
	}
}

func (i *Instance) closeDone() { i.doneOnce.Do(func() { close(i.done) }) }

func (i *Instance) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("POST /v1/chat/completions", i.chat)
	mux.HandleFunc("POST /v1/completions", i.chat)
	mux.HandleFunc("POST /v1/embeddings", i.embeddings)
	return mux
}

func (i *Instance) track() func() {
	i.requests.Add(1)
	n := i.inflight.Add(1)
	for {
		p := i.peak.Load()
		if n <= p || i.peak.CompareAndSwap(p, n) {
			break
		}
	}
	return func() { i.inflight.Add(-1) }
}

func (i *Instance) chat(w http.ResponseWriter, r *http.Request) {
	defer i.track()()
	var req struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)

	if !req.Stream {
		select {
		case <-time.After(i.delay):
		case <-r.Context().Done():
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "fake-" + fmt.Sprint(i.requests.Load()), "object": "chat.completion",
			"model": i.spec.ID,
			"choices": []any{map[string]any{"index": 0, "finish_reason": "stop",
				"message": map[string]any{"role": "assistant", "content": "ok from " + i.spec.ID}}},
			"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
		})
		return
	}

	// Stream 5 tokens spread across the delay.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	fl, _ := w.(http.Flusher)
	step := i.delay / 5
	for n := 0; n < 5; n++ {
		select {
		case <-time.After(step):
		case <-r.Context().Done():
			return
		}
		chunk, _ := json.Marshal(map[string]any{
			"id": "fake", "object": "chat.completion.chunk", "model": i.spec.ID,
			"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": fmt.Sprintf("tok%d ", n)}}},
		})
		fmt.Fprintf(w, "data: %s\n\n", chunk)
		if fl != nil {
			fl.Flush()
		}
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	if fl != nil {
		fl.Flush()
	}
}

func (i *Instance) embeddings(w http.ResponseWriter, r *http.Request) {
	defer i.track()()
	var req struct {
		Input json.RawMessage `json:"input"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)

	// input may be a string or an array of strings.
	n := 1
	var arr []json.RawMessage
	if json.Unmarshal(req.Input, &arr) == nil {
		n = len(arr)
	}

	select {
	case <-time.After(i.delay):
	case <-r.Context().Done():
		return
	}
	data := make([]any, n)
	for k := range data {
		data[k] = map[string]any{"object": "embedding", "index": k, "embedding": []float64{0, 0, 0, 0, 0, 0, 0, float64(k)}}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list", "model": i.spec.ID, "data": data,
		"usage": map[string]any{"prompt_tokens": n * 8, "total_tokens": n * 8},
	})
}
