// Package llamacpp runs models as llama-server processes, one per model.
//
// Load spawns `llama-server --model ... --port N` in its own process group,
// tees its output to a log file and polls GET /health until it answers 200.
// Stop sends SIGTERM to the group and escalates to SIGKILL after
// runtime.StopGrace. No VRAM knowledge lives here: the scheduler measures
// the process through the GPU monitor by PID.
package llamacpp

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gridcore/gridcore/internal/config"
	"github.com/gridcore/gridcore/internal/model"
	"github.com/gridcore/gridcore/internal/runtime"
)

// Runtime implements runtime.Runtime for llama.cpp.
type Runtime struct {
	name        string
	binary      string
	defaultArgs []string
	loadTimeout time.Duration
	logDir      string

	// HealthInterval is the poll period while waiting for readiness.
	HealthInterval time.Duration
	// Env is appended to the child environment (e.g. CUDA_VISIBLE_DEVICES).
	Env []string
	// Registry, when set, records live children for orphan cleanup.
	Registry *Registry
	// tailLines is how many log lines are attached to a load error.
	tailLines int
}

// New creates a runtime from its config entry. logDir receives one file per
// instance: <model>-<port>.log.
func New(name string, rc config.Runtime, logDir string) *Runtime {
	return &Runtime{
		name:           name,
		binary:         rc.Binary,
		defaultArgs:    append([]string(nil), rc.DefaultArgs...),
		loadTimeout:    rc.LoadTimeout,
		logDir:         logDir,
		HealthInterval: 200 * time.Millisecond,
		tailLines:      50,
	}
}

func (r *Runtime) Name() string { return r.name }

// Args builds the llama-server command line for spec on port. Exposed for
// tests and `gridcore models --args`.
func (r *Runtime) Args(spec *model.Spec, port int) []string {
	args := []string{
		"--model", spec.Path,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--ctx-size", strconv.Itoa(spec.Ctx),
		"--parallel", strconv.Itoa(spec.Parallel),
		"--alias", spec.ID,
		"--no-webui",
	}
	if !hasFlag(spec.Args, "-ngl", "--gpu-layers", "--n-gpu-layers") && !hasFlag(r.defaultArgs, "-ngl", "--gpu-layers", "--n-gpu-layers") {
		args = append(args, "--n-gpu-layers", "999")
	}
	if spec.MMProj != "" {
		args = append(args, "--mmproj", spec.MMProj)
	} else {
		// Newer llama-server auto-downloads mmproj for known VL models; we
		// never want network access from the runtime.
		args = append(args, "--no-mmproj-auto")
	}
	if spec.HasCapability(config.CapEmbedding) && !spec.HasCapability(config.CapChat) && !spec.HasCapability(config.CapCompletion) {
		args = append(args, "--embedding")
	}
	args = append(args, r.defaultArgs...)
	args = append(args, spec.Args...)
	return args
}

func hasFlag(args []string, names ...string) bool {
	for _, a := range args {
		for _, n := range names {
			if a == n || strings.HasPrefix(a, n+"=") {
				return true
			}
		}
	}
	return false
}

// Load implements runtime.Runtime.
func (r *Runtime) Load(ctx context.Context, spec *model.Spec, port int) (runtime.Instance, error) {
	if r.loadTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.loadTimeout)
		defer cancel()
	}

	var logFile *os.File
	if r.logDir != "" {
		if err := os.MkdirAll(r.logDir, 0o755); err != nil {
			return nil, fmt.Errorf("llamacpp: log dir: %w", err)
		}
		name := filepath.Join(r.logDir, fmt.Sprintf("%s-%d.log", sanitize(spec.ID), port))
		f, err := os.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			return nil, fmt.Errorf("llamacpp: open log: %w", err)
		}
		logFile = f
	}

	args := r.Args(spec, port)
	cmd := exec.Command(r.binary, args...)
	cmd.Env = append(os.Environ(), r.Env...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	// Do not tie the child's lifetime to ctx: ctx bounds the *load*, and a
	// timed-out load is killed explicitly below.

	tail := newTail(r.tailLines)
	var sinks []io.Writer = []io.Writer{tail}
	if logFile != nil {
		sinks = append(sinks, logFile)
	}
	pr, pw := io.Pipe()
	cmd.Stdout = pw
	cmd.Stderr = pw

	if err := cmd.Start(); err != nil {
		pw.Close()
		if logFile != nil {
			logFile.Close()
		}
		return nil, fmt.Errorf("llamacpp: start %s: %w", r.binary, err)
	}

	inst := &Instance{
		spec:    spec,
		addr:    "127.0.0.1:" + strconv.Itoa(port),
		port:    port,
		cmd:     cmd,
		done:    make(chan struct{}),
		tail:    tail,
		logPath: pathOf(logFile),
		client:  &http.Client{Timeout: 2 * time.Second},
	}
	if r.Registry != nil {
		r.Registry.Add(RegistryEntry{PID: cmd.Process.Pid, Model: spec.ID, Port: port, Binary: r.binary, StartedAt: time.Now()})
	}

	// Copy child output to the sinks until the pipe closes.
	copyDone := make(chan struct{})
	go func() {
		defer close(copyDone)
		_, _ = io.Copy(io.MultiWriter(sinks...), pr)
		if logFile != nil {
			logFile.Close()
		}
	}()
	// Reap the process; closing pw ends the copier.
	go func() {
		err := cmd.Wait()
		pw.Close()
		<-copyDone
		inst.setErr(err)
		if r.Registry != nil {
			r.Registry.Remove(cmd.Process.Pid)
		}
		close(inst.done)
	}()

	// Wait for readiness.
	tick := time.NewTicker(r.HealthInterval)
	defer tick.Stop()
	for {
		select {
		case <-inst.done:
			return nil, &LoadError{Model: spec.ID, Cause: fmt.Errorf("exited during load: %v", inst.Err()), tail: tail.String(), Summary: tail.Summary()}
		case <-ctx.Done():
			_ = inst.Stop(context.Background())
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				return nil, &LoadError{Model: spec.ID, Cause: runtime.ErrLoadTimeout, tail: tail.String(), Summary: tail.Summary()}
			}
			return nil, ctx.Err()
		case <-tick.C:
			if inst.Health(ctx) == nil {
				inst.started = time.Now()
				return inst, nil
			}
		}
	}
}

// LoadError is a failed load with the server's output attached. Error()
// stays short (one summary line) so it can be shown to API clients; Tail()
// has the full context for logs.
type LoadError struct {
	Model   string
	Cause   error
	Summary string // most relevant output line (last error, else last line)
	tail    string
}

func (e *LoadError) Error() string {
	if e.Summary != "" {
		return fmt.Sprintf("llamacpp: %s %v: %s", e.Model, e.Cause, e.Summary)
	}
	return fmt.Sprintf("llamacpp: %s %v", e.Model, e.Cause)
}

func (e *LoadError) Unwrap() error { return e.Cause }

// Tail returns the last lines of llama-server output.
func (e *LoadError) Tail() string { return e.tail }

// OOM reports whether the output indicates the device ran out of memory.
func (e *LoadError) OOM() bool { return looksLikeOOM(e.Summary) || looksLikeOOM(e.tail) }

func looksLikeOOM(s string) bool {
	l := strings.ToLower(s)
	return strings.Contains(l, "out of memory") ||
		(strings.Contains(l, "failed to allocate") && (strings.Contains(l, "cuda") || strings.Contains(l, "buffer"))) ||
		strings.Contains(l, "unable to allocate")
}

// Instance is one llama-server process.
type Instance struct {
	spec    *model.Spec
	addr    string
	port    int
	cmd     *exec.Cmd
	started time.Time
	tail    *tail
	logPath string
	client  *http.Client

	done    chan struct{}
	errMu   sync.Mutex
	err     error
	stopMu  sync.Mutex
	stopped bool
}

func (i *Instance) Spec() *model.Spec     { return i.spec }
func (i *Instance) Addr() string          { return i.addr }
func (i *Instance) PID() int              { return i.cmd.Process.Pid }
func (i *Instance) StartedAt() time.Time  { return i.started }
func (i *Instance) Done() <-chan struct{} { return i.done }

// LogPath is the file receiving the process output ("" if logging is off).
func (i *Instance) LogPath() string { return i.logPath }

func (i *Instance) Err() error {
	i.errMu.Lock()
	defer i.errMu.Unlock()
	return i.err
}

func (i *Instance) setErr(err error) {
	i.errMu.Lock()
	defer i.errMu.Unlock()
	i.err = err
}

// Health implements runtime.Instance. llama-server answers 503 while the
// model is still loading and 200 once slots are available.
func (i *Instance) Health(ctx context.Context) error {
	select {
	case <-i.done:
		return errors.New("llamacpp: process exited")
	default:
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+i.addr+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := i.client.Do(req)
	if err != nil {
		return err
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("llamacpp: health %d", resp.StatusCode)
	}
	return nil
}

// Stop implements runtime.Instance: SIGTERM to the process group, then
// SIGKILL after runtime.StopGrace. Safe to call more than once.
func (i *Instance) Stop(ctx context.Context) error {
	i.stopMu.Lock()
	already := i.stopped
	i.stopped = true
	i.stopMu.Unlock()

	select {
	case <-i.done:
		return nil
	default:
	}
	pgid := -i.cmd.Process.Pid // negative pid = whole group (Setpgid)
	if !already {
		if err := syscall.Kill(pgid, syscall.SIGTERM); err != nil && !errors.Is(err, syscall.ESRCH) {
			return fmt.Errorf("llamacpp: SIGTERM: %w", err)
		}
	}
	grace := time.NewTimer(runtime.StopGrace)
	defer grace.Stop()
	select {
	case <-i.done:
		return nil
	case <-grace.C:
	case <-ctx.Done():
	}
	_ = syscall.Kill(pgid, syscall.SIGKILL)
	select {
	case <-i.done:
		return nil
	case <-time.After(5 * time.Second):
		return errors.New("llamacpp: process did not exit after SIGKILL")
	}
}

// ---- helpers ----

// tail keeps the last n lines of the child's output for error messages.
type tail struct {
	mu    sync.Mutex
	n     int
	lines []string
	buf   []byte
}

func newTail(n int) *tail { return &tail{n: n} }

func (t *tail) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, p...)
	for {
		i := strings.IndexByte(string(t.buf), '\n')
		if i < 0 {
			break
		}
		t.push(string(t.buf[:i]))
		t.buf = t.buf[i+1:]
	}
	return len(p), nil
}

func (t *tail) push(line string) {
	t.lines = append(t.lines, line)
	if len(t.lines) > t.n {
		t.lines = t.lines[len(t.lines)-t.n:]
	}
}

// Summary picks the most informative line: the last one llama-server
// logged at error level, otherwise the last non-empty line.
func (t *tail) Summary() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	lines := append([]string(nil), t.lines...)
	if len(t.buf) > 0 {
		lines = append(lines, string(t.buf))
	}
	var last string
	for i := len(lines) - 1; i >= 0; i-- {
		l := strings.TrimSpace(lines[i])
		if l == "" {
			continue
		}
		if last == "" {
			last = l
		}
		// llama.cpp log format: "<time> E <component> message"
		if f := strings.Fields(l); len(f) > 2 && f[1] == "E" {
			return strings.TrimSpace(strings.Join(f[2:], " "))
		}
		if strings.Contains(strings.ToLower(l), "error") {
			return l
		}
	}
	return last
}

func (t *tail) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	lines := append([]string(nil), t.lines...)
	if len(t.buf) > 0 {
		lines = append(lines, string(t.buf))
	}
	if len(lines) == 0 {
		return "(no output)"
	}
	var b strings.Builder
	w := bufio.NewWriter(&b)
	fmt.Fprintf(w, "--- last %d lines of llama-server output ---\n", len(lines))
	for _, l := range lines {
		fmt.Fprintln(w, l)
	}
	w.Flush()
	return b.String()
}

func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			return r
		}
		return '_'
	}, s)
}

func pathOf(f *os.File) string {
	if f == nil {
		return ""
	}
	return f.Name()
}
