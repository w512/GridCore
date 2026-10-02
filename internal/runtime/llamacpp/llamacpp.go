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
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/w512/gridcore/internal/config"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/runtime"
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

	versionOnce sync.Once
	version     string
	helpOnce    sync.Once
	help        string
}

var errWarmupOOM = errors.New("device out of memory during warmup")

// unifiedDefault is one launch default for unified memory: the flag pairs
// to add when the binary knows them (the first spelling it knows wins), and
// the flags that mean the user already chose.
type unifiedDefault struct {
	options [][]string
	user    []string
}

// unifiedDefaults keep a model's footprint what it is right after loading,
// so that admission can rely on it. Measured on an M4 Pro, llama.cpp
// b11146, gemma4-12b at 16K x 2 slots:
//
//   - without mmap the weights are anonymous memory: they show up in the
//     footprint, leave with the process, and Metal does not wrap the whole
//     file (host-side tensors included) in one GPU buffer; it loads faster
//     too (3.65 s against 6.8);
//   - the RAM prompt cache (--cache-ram, default 8192 MiB) grew the process
//     from 8.8 to 18.2 GB over 12 distinct prompts;
//   - context checkpoints (up to 32 per slot) added 1.1 GB more.
//
// On a dedicated GPU all of that is host RAM and none of GridCore's
// business; on unified memory it is the same pool as the model. A user who
// wants the cache or checkpoints back sets the flag in the model's args,
// and the measured peak then covers the growth.
var unifiedDefaults = []unifiedDefault{
	{options: [][]string{{"--load-mode", "none"}, {"--no-mmap"}}, user: []string{"--load-mode", "-lm", "--mmap", "--no-mmap", "--mlock"}},
	{options: [][]string{{"--cache-ram", "0"}}, user: []string{"--cache-ram", "-cram"}},
	{options: [][]string{{"--ctx-checkpoints", "0"}}, user: []string{"--ctx-checkpoints", "-ctxcp", "--swa-checkpoints"}},
}

// unifiedArgs returns the unified-memory defaults the binary understands and
// the model's args do not override.
func (r *Runtime) unifiedArgs(spec *model.Spec) []string {
	help := r.helpText()
	var out []string
	for _, d := range unifiedDefaults {
		if hasFlag(spec.Args, d.user...) || hasFlag(r.defaultArgs, d.user...) {
			continue
		}
		for _, opt := range d.options {
			if strings.Contains(help, opt[0]) {
				out = append(out, opt...)
				break
			}
		}
	}
	return out
}

// helpText is `llama-server --help`, asked once: flags come and go between
// builds (--no-mmap became --load-mode in b11146).
func (r *Runtime) helpText() string {
	r.helpOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), versionTimeout)
		defer cancel()
		cmd := exec.CommandContext(ctx, r.binary, "--help")
		cmd.Env = append(os.Environ(), r.Env...)
		cmd.WaitDelay = time.Second
		out, _ := cmd.CombinedOutput()
		r.help = string(out)
	})
	return r.help
}

// versionTimeout bounds `llama-server --version`. A CUDA build initialises
// the driver to list devices first, which takes a fraction of a second.
const versionTimeout = 5 * time.Second

// Version implements runtime.Versioned: "b<build>-<commit>", plus the GPU
// backends the binary loaded ("b11060-a1b2c3d+CUDA"). The binary is asked
// once; "" if it does not answer in time or prints no version line.
func (r *Runtime) Version() string {
	r.versionOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), versionTimeout)
		defer cancel()
		cmd := exec.CommandContext(ctx, r.binary, "--version")
		cmd.Env = append(os.Environ(), r.Env...)
		cmd.WaitDelay = time.Second
		out, _ := cmd.CombinedOutput() // the exit status says nothing useful
		r.version = parseVersion(string(out))
	})
	return r.version
}

var (
	// "version: 11060 (a1b2c3d)" (older builds) or
	// "version: 0.4.1-dev (build 11060, commit 426090367)" (b11060).
	versionLine = regexp.MustCompile(`(?m)^version:\s*(?:(\d+)\s*\(([0-9a-f]+)\)|\S+\s*\(build\s+(\d+),\s*commit\s+([0-9a-f]+)\))`)
	backendLine = regexp.MustCompile(`(?m)^load_backend: loaded (\S+) backend`)
)

// parseVersion extracts the build identity from `llama-server --version`:
//
//	0.00.000.204 I srv  llama_server: initializing ...
//	version: 0.4.1-dev (build 11060, commit 426090367)
//	built with GNU 13.3.0 for Linux x86_64
//
// Builds with dynamically loaded backends also print "load_backend: loaded
// CUDA backend from ..."; the CPU backend is always present and left out.
func parseVersion(out string) string {
	m := versionLine.FindStringSubmatch(out)
	if m == nil {
		return ""
	}
	build, commit := m[1], m[2]
	if build == "" {
		build, commit = m[3], m[4]
	}
	id := "b" + build + "-" + commit
	seen := map[string]bool{}
	var backends []string
	for _, b := range backendLine.FindAllStringSubmatch(out, -1) {
		if name := b[1]; name != "CPU" && !seen[name] {
			seen[name] = true
			backends = append(backends, name)
		}
	}
	sort.Strings(backends)
	if len(backends) > 0 {
		id += "+" + strings.Join(backends, "+")
	}
	return id
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
	if spec.Unified {
		args = append(args, r.unifiedArgs(spec)...)
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
				// Metal reports an overcommitted device by failing the
				// warmup run, yet the server still comes up and answers
				// /health; every request (and the neighbours') then fails.
				if tail.OOMs() > 0 {
					_ = inst.Stop(context.Background())
					return nil, &LoadError{Model: spec.ID, Cause: errWarmupOOM, tail: tail.String(), Summary: tail.Summary()}
				}
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

// looksLikeOOM matches CUDA allocation failures and Metal's
// "error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)".
func looksLikeOOM(s string) bool {
	l := strings.ToLower(s)
	return strings.Contains(l, "out of memory") ||
		(strings.Contains(l, "failed to allocate") && (strings.Contains(l, "cuda") || strings.Contains(l, "buffer"))) ||
		strings.Contains(l, "unable to allocate") ||
		strings.Contains(l, "insufficient memory") ||
		strings.Contains(l, "erroroutofmemory")
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

// OOMs implements runtime.OOMReporter.
func (i *Instance) OOMs() int { return i.tail.OOMs() }

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
	ooms  int // out-of-memory lines seen, over the whole output
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
	if looksLikeOOM(line) {
		t.ooms++
	}
	t.lines = append(t.lines, line)
	if len(t.lines) > t.n {
		t.lines = t.lines[len(t.lines)-t.n:]
	}
}

// OOMs counts out-of-memory lines in the output so far.
func (t *tail) OOMs() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.ooms
}

// Summary picks the most informative line: the first out-of-memory error
// (Metal logs "command buffer failed" first and the reason on the next
// line), else the first line llama-server logged at error level (the root
// cause; later ones are consequences), otherwise the last line mentioning an
// error, otherwise the last line.
func (t *tail) Summary() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	lines := append([]string(nil), t.lines...)
	if len(t.buf) > 0 {
		lines = append(lines, string(t.buf))
	}
	for _, raw := range lines {
		if looksLikeOOM(raw) {
			if f := strings.Fields(strings.TrimSpace(raw)); len(f) > 2 && f[1] == "E" {
				return strings.Join(f[2:], " ")
			}
			return strings.TrimSpace(raw)
		}
	}
	var last, lastErr string
	for _, raw := range lines {
		l := strings.TrimSpace(raw)
		if l == "" {
			continue
		}
		last = l
		// llama.cpp log format: "<time> E <component> message"
		if f := strings.Fields(l); len(f) > 2 && f[1] == "E" {
			return strings.TrimSpace(strings.Join(f[2:], " "))
		}
		if strings.Contains(strings.ToLower(l), "error") {
			lastErr = l
		}
	}
	if lastErr != "" {
		return lastErr
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
