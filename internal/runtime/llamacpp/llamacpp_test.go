package llamacpp

import (
	"context"
	"errors"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/config"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/runtime"
)

// stubServer is a tiny Python HTTP server standing in for llama-server. It
// parses --port, optionally sleeps before becoming healthy (STUB_DELAY), can
// exit immediately (STUB_CRASH) and ignores SIGTERM (STUB_IGNORE_TERM) so
// the SIGKILL path is exercised. --help prints STUB_HELP. Like Metal on an
// overcommitted Mac it can fail its warmup and still come up
// (STUB_WARMUP_OOM), or start failing STUB_OOM_AFTER seconds after start.
const stubServer = `#!/usr/bin/env python3
import http.server, os, signal, sys, threading, time
OOM = "0.27.348586 E error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
def oom():
    print("0.27.348578 E ggml_metal_synchronize: error: command buffer 0 failed with status 5", flush=True)
    print(OOM, flush=True)
if "--help" in sys.argv:
    print(os.environ.get("STUB_HELP", "-lm,   --load-mode MODE   model loading mode (default: auto)"))
    sys.exit(0)
port = int(sys.argv[sys.argv.index("--port") + 1])
print("stub llama-server starting on", port, "args:", " ".join(sys.argv[1:]), flush=True)
if os.environ.get("STUB_CRASH"):
    print("fatal: pretend the model file is corrupt", file=sys.stderr, flush=True)
    sys.exit(3)
if os.environ.get("STUB_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, lambda *a: print("ignoring SIGTERM", flush=True))
time.sleep(float(os.environ.get("STUB_DELAY", "0")))
if os.environ.get("STUB_WARMUP_OOM"):
    oom()
if os.environ.get("STUB_OOM_AFTER"):
    threading.Timer(float(os.environ["STUB_OOM_AFTER"]), oom).start()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer.allow_reuse_address = True
srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
print("stub ready", flush=True)
srv.serve_forever()
`

func writeStub(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}
	p := filepath.Join(t.TempDir(), "llama-server")
	if err := os.WriteFile(p, []byte(stubServer), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

func newRT(t *testing.T, env ...string) (*Runtime, string) {
	t.Helper()
	logDir := filepath.Join(t.TempDir(), "logs")
	rt := New("llamacpp", config.Runtime{
		Type: config.RuntimeLlamaCpp, Binary: writeStub(t),
		LoadTimeout: 5 * time.Second, DefaultArgs: []string{"--flash-attn", "on"},
	}, logDir)
	rt.HealthInterval = 20 * time.Millisecond
	rt.Env = env
	return rt, logDir
}

func spec() *model.Spec {
	return &model.Spec{ID: "qwen3-14b", Runtime: "llamacpp", RuntimeType: config.RuntimeLlamaCpp,
		Path: "/models/q.gguf", Capabilities: []string{"chat"}, Ctx: 8192, Parallel: 2, Args: []string{"--jinja"}}
}

func TestArgs(t *testing.T) {
	rt, _ := newRT(t)
	got := strings.Join(rt.Args(spec(), 41000), " ")
	for _, want := range []string{
		"--model /models/q.gguf", "--host 127.0.0.1", "--port 41000", "--ctx-size 8192", "--parallel 2",
		"--alias qwen3-14b", "--no-webui", "--n-gpu-layers 999", "--no-mmproj-auto", "--flash-attn on", "--jinja",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("args missing %q: %s", want, got)
		}
	}
	if strings.Contains(got, "--embedding") {
		t.Error("chat model must not get --embedding")
	}

	emb := &model.Spec{ID: "e", Path: "/m/e.gguf", Capabilities: []string{"embedding"}, Ctx: 2048, Parallel: 4}
	got = strings.Join(rt.Args(emb, 1), " ")
	if !strings.Contains(got, "--embedding") {
		t.Errorf("embedding-only model must get --embedding: %s", got)
	}

	vl := &model.Spec{ID: "vl", Path: "/m/vl.gguf", MMProj: "/m/mmproj.gguf", Capabilities: []string{"chat", "vision"}, Ctx: 4096, Parallel: 1}
	got = strings.Join(rt.Args(vl, 1), " ")
	if !strings.Contains(got, "--mmproj /m/mmproj.gguf") || strings.Contains(got, "--no-mmproj-auto") {
		t.Errorf("vision args: %s", got)
	}

	custom := spec()
	custom.Args = []string{"-ngl", "20"}
	got = strings.Join(rt.Args(custom, 1), " ")
	if strings.Contains(got, "--n-gpu-layers 999") || !strings.Contains(got, "-ngl 20") {
		t.Errorf("user -ngl must win: %s", got)
	}
}

func TestArgsUnifiedLoadMode(t *testing.T) {
	rt, _ := newRT(t)
	sp := spec()
	sp.Unified = true
	if got := strings.Join(rt.Args(sp, 1), " "); !strings.Contains(got, "--load-mode none") {
		t.Errorf("unified memory must load without mmap: %s", got)
	}
	if got := strings.Join(rt.Args(spec(), 1), " "); strings.Contains(got, "--load-mode") {
		t.Errorf("dedicated GPUs keep the default: %s", got)
	}
	sp.Args = []string{"--load-mode", "mmap"}
	if got := strings.Join(rt.Args(sp, 1), " "); strings.Count(got, "--load-mode") != 1 {
		t.Errorf("the user's load mode must win: %s", got)
	}

	old, _ := newRT(t, "STUB_HELP=--no-mmap   do not memory-map model")
	sp.Args = nil
	if got := strings.Join(old.Args(sp, 1), " "); !strings.Contains(got, "--no-mmap") || strings.Contains(got, "--load-mode") {
		t.Errorf("builds before --load-mode get --no-mmap: %s", got)
	}
	neither, _ := newRT(t, "STUB_HELP=usage: llama-server [options]")
	if got := strings.Join(neither.Args(sp, 1), " "); strings.Contains(got, "mmap") || strings.Contains(got, "load-mode") {
		t.Errorf("a build with neither flag gets none: %s", got)
	}
}

// On an overcommitted Mac the warmup fails but the server still answers
// /health; the load must fail as OOM and leave no process behind.
func TestLoadWarmupOOM(t *testing.T) {
	rt, _ := newRT(t, "STUB_WARMUP_OOM=1")
	_, err := rt.Load(context.Background(), spec(), freePort(t))
	var le *LoadError
	if !errors.As(err, &le) || !errors.Is(err, errWarmupOOM) || !le.OOM() {
		t.Fatalf("want a warmup OOM load error, got %v", err)
	}
	if !strings.Contains(le.Summary, "Insufficient Memory") {
		t.Errorf("summary should name the cause: %q", le.Summary)
	}
}

func TestInstanceCountsOOMs(t *testing.T) {
	rt, _ := newRT(t, "STUB_OOM_AFTER=0.3")
	inst, err := rt.Load(context.Background(), spec(), freePort(t))
	if err != nil {
		t.Fatal(err)
	}
	defer inst.Stop(context.Background())
	var rep runtime.OOMReporter = inst.(*Instance)
	if rep.OOMs() != 0 {
		t.Fatalf("no OOM yet, got %d", rep.OOMs())
	}
	deadline := time.Now().Add(3 * time.Second)
	for rep.OOMs() == 0 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if rep.OOMs() != 1 {
		t.Errorf("one Metal OOM (two log lines) should count once, got %d", rep.OOMs())
	}
}

func TestLoadHealthStop(t *testing.T) {
	rt, logDir := newRT(t)
	ctx := context.Background()
	inst, err := rt.Load(ctx, spec(), freePort(t))
	if err != nil {
		t.Fatal(err)
	}
	if inst.PID() <= 0 || !strings.HasPrefix(inst.Addr(), "127.0.0.1:") {
		t.Errorf("pid=%d addr=%s", inst.PID(), inst.Addr())
	}
	if err := inst.Health(ctx); err != nil {
		t.Errorf("health: %v", err)
	}
	// Process group: child must be its own group leader.
	pgid, err := syscall.Getpgid(inst.PID())
	if err != nil || pgid != inst.PID() {
		t.Errorf("pgid=%d pid=%d err=%v", pgid, inst.PID(), err)
	}
	logs, _ := filepath.Glob(filepath.Join(logDir, "qwen3-14b-*.log"))
	if len(logs) != 1 {
		t.Fatalf("expected one log file, got %v", logs)
	}

	start := time.Now()
	if err := inst.Stop(ctx); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if d := time.Since(start); d > 2*time.Second {
		t.Errorf("SIGTERM stop took %v; expected fast exit", d)
	}
	select {
	case <-inst.Done():
	default:
		t.Error("Done should be closed after Stop")
	}
	if inst.Health(ctx) == nil {
		t.Error("health must fail after stop")
	}
	if err := inst.Stop(ctx); err != nil {
		t.Errorf("second Stop must be a no-op: %v", err)
	}
	// Process is really gone.
	if err := syscall.Kill(inst.PID(), 0); err == nil {
		t.Error("process still alive after Stop")
	}
	raw, _ := os.ReadFile(logs[0])
	if !strings.Contains(string(raw), "stub ready") {
		t.Errorf("log file should capture child output, got: %s", raw)
	}
}

func TestLoadCrashIncludesTail(t *testing.T) {
	rt, _ := newRT(t, "STUB_CRASH=1")
	_, err := rt.Load(context.Background(), spec(), freePort(t))
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "exited during load") || !strings.Contains(err.Error(), "pretend the model file is corrupt") {
		t.Errorf("error should include exit info and the summary line:\n%v", err)
	}
	var le *LoadError
	if !errors.As(err, &le) {
		t.Fatalf("want *LoadError, got %T", err)
	}
	if !strings.Contains(le.Tail(), "stub llama-server starting") {
		t.Errorf("tail should hold full output: %q", le.Tail())
	}
	if strings.Count(err.Error(), "\n") > 0 {
		t.Errorf("Error() must be a single line for API clients: %q", err.Error())
	}
	if le.OOM() {
		t.Error("corrupt-file failure must not look like OOM")
	}
}

func TestSummaryAndOOM(t *testing.T) {
	tl := newTail(10)
	tl.Write([]byte("0.00.1 I srv init\n0.00.2 E alloc_tensor_range: failed to allocate CUDA0 buffer of size 8558218240\n0.00.3 E llama_model_load: error loading model: unable to allocate CUDA0 buffer\n0.00.4 I srv exiting\n"))
	sum := tl.Summary()
	if !strings.HasPrefix(sum, "alloc_tensor_range: failed to allocate CUDA0 buffer") {
		t.Errorf("summary should be the first E line (root cause), got %q", sum)
	}
	le := &LoadError{Model: "m", Cause: errors.New("exit 1"), Summary: sum, tail: tl.String()}
	if !le.OOM() {
		t.Error("CUDA allocation failure must be detected as OOM")
	}
	plain := newTail(3)
	plain.Write([]byte("hello\nworld\n"))
	if plain.Summary() != "world" {
		t.Errorf("fallback summary = %q", plain.Summary())
	}
}

func TestLoadTimeoutKillsProcess(t *testing.T) {
	rt, _ := newRT(t, "STUB_DELAY=10")
	rt.loadTimeout = 300 * time.Millisecond
	port := freePort(t)
	_, err := rt.Load(context.Background(), spec(), port)
	if !errors.Is(err, runtime.ErrLoadTimeout) {
		t.Fatalf("want ErrLoadTimeout, got %v", err)
	}
	// Port must be free again: the child was killed.
	ln, lerr := net.Listen("tcp", net.JoinHostPort("127.0.0.1", itoa(port)))
	if lerr != nil {
		t.Fatalf("port still held after timeout: %v", lerr)
	}
	ln.Close()
}

func TestLoadCancelled(t *testing.T) {
	rt, _ := newRT(t, "STUB_DELAY=10")
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(50 * time.Millisecond); cancel() }()
	_, err := rt.Load(ctx, spec(), freePort(t))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

func TestStopEscalatesToSIGKILL(t *testing.T) {
	rt, _ := newRT(t, "STUB_IGNORE_TERM=1")
	inst, err := rt.Load(context.Background(), spec(), freePort(t))
	if err != nil {
		t.Fatal(err)
	}
	// Use a short context so the test does not wait the full StopGrace.
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	start := time.Now()
	if err := inst.Stop(ctx); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if d := time.Since(start); d > 3*time.Second {
		t.Errorf("SIGKILL escalation took %v", d)
	}
	if err := syscall.Kill(inst.PID(), 0); err == nil {
		t.Error("process survived SIGKILL")
	}
}

func TestExternalDeathClosesDone(t *testing.T) {
	rt, _ := newRT(t)
	inst, err := rt.Load(context.Background(), spec(), freePort(t))
	if err != nil {
		t.Fatal(err)
	}
	_ = syscall.Kill(inst.PID(), syscall.SIGKILL)
	select {
	case <-inst.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("Done not closed after external kill")
	}
	if inst.Err() == nil {
		t.Error("Err should report the abnormal exit")
	}
}

func TestMissingBinary(t *testing.T) {
	rt := New("llamacpp", config.Runtime{Binary: "/nonexistent/llama-server", LoadTimeout: time.Second}, "")
	_, err := rt.Load(context.Background(), spec(), freePort(t))
	if err == nil || !strings.Contains(err.Error(), "start") {
		t.Fatalf("expected start error, got %v", err)
	}
}

func TestTail(t *testing.T) {
	tl := newTail(3)
	tl.Write([]byte("a\nb\nc\nd\npartial"))
	s := tl.String()
	if strings.Contains(s, "a\n") || !strings.Contains(s, "d\n") || !strings.Contains(s, "partial") {
		t.Errorf("tail = %q", s)
	}
	if newTail(3).String() != "(no output)" {
		t.Error("empty tail")
	}
}

func itoa(i int) string { return strconv.Itoa(i) }

func TestRegistryReapsOrphans(t *testing.T) {
	regPath := filepath.Join(t.TempDir(), "instances.json")
	reg, err := OpenRegistry(regPath)
	if err != nil {
		t.Fatal(err)
	}
	rt, _ := newRT(t)
	rt.Registry = reg
	inst, err := rt.Load(context.Background(), spec(), freePort(t))
	if err != nil {
		t.Fatal(err)
	}
	pid := inst.PID()

	// Simulate a GridCore crash: forget the instance object, reopen the
	// registry from disk and reap.
	reg2, err := OpenRegistry(regPath)
	if err != nil {
		t.Fatal(err)
	}
	reaped := reg2.ReapOrphans()
	if len(reaped) != 1 || reaped[0].PID != pid || reaped[0].Model != "qwen3-14b" {
		t.Fatalf("reaped = %+v", reaped)
	}
	select {
	case <-inst.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("orphan not killed")
	}
	reg3, _ := OpenRegistry(regPath)
	if got := reg3.ReapOrphans(); len(got) != 0 {
		t.Errorf("registry should be empty after reap, got %+v", got)
	}

	// Normal exit removes the entry.
	inst2, err := rt.Load(context.Background(), spec(), freePort(t))
	if err != nil {
		t.Fatal(err)
	}
	_ = inst2.Stop(context.Background())
	reg4, _ := OpenRegistry(regPath)
	if got := reg4.ReapOrphans(); len(got) != 0 {
		t.Errorf("stopped instance must not be in the registry, got %+v", got)
	}
}

func TestLooksLikeOurs(t *testing.T) {
	e := RegistryEntry{Binary: "/opt/llama.cpp/current/llama-server", Port: 41003}
	cases := []struct {
		argv []string
		want bool
	}{
		{[]string{"/opt/llama.cpp/current/llama-server", "--model", "m.gguf", "--port", "41003"}, true},
		{[]string{"/opt/llama.cpp/llama-b11060/llama-server", "--port=41003"}, true},   // symlink resolved by the kernel
		{[]string{"/usr/bin/python3", "/tmp/x/llama-server", "--port", "41003"}, true}, // interpreter + script
		{[]string{"/opt/llama.cpp/current/llama-server", "--port", "41004"}, false},    // other port
		{[]string{"/usr/bin/firefox", "--port", "41003"}, false},                       // reused pid
		{nil, false},
	}
	for _, c := range cases {
		if got := looksLikeOurs(c.argv, e); got != c.want {
			t.Errorf("looksLikeOurs(%v) = %v, want %v", c.argv, got, c.want)
		}
	}
}

// A recorded PID that now belongs to an unrelated process must survive the
// reap. macOS has no /proc, so this exercises the ps fallback there.
func TestReapSkipsReusedPID(t *testing.T) {
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cmd.Process.Kill(); _ = cmd.Wait() })
	if argv := procArgv(cmd.Process.Pid); len(argv) < 2 || filepath.Base(argv[0]) != "sleep" || argv[1] != "30" {
		t.Fatalf("procArgv = %q, want sleep 30", argv)
	}

	regPath := filepath.Join(t.TempDir(), "instances.json")
	reg, _ := OpenRegistry(regPath)
	reg.Add(RegistryEntry{PID: cmd.Process.Pid, Model: "m", Port: 41003, Binary: "/opt/llama.cpp/current/llama-server"})
	reg2, _ := OpenRegistry(regPath)
	if got := reg2.ReapOrphans(); len(got) != 0 {
		t.Fatalf("reaped an unrelated process: %+v", got)
	}
	if !alive(cmd.Process.Pid) {
		t.Fatal("unrelated process was killed")
	}
}

func TestParseVersion(t *testing.T) {
	cases := []struct{ out, want string }{
		// Verbatim from the prebuilt CUDA release b11060 on the 4060 Ti box.
		{"0.00.000.204 I srv  llama_server: initializing ...\nversion: 0.4.1-dev (build 11060, commit 426090367)\nbuilt with GNU 13.3.0 for Linux x86_64\n", "b11060-426090367"},
		{"version: 11060 (a1b2c3d)\nbuilt with cc 13.3.0 for x86_64-linux-gnu\n", "b11060-a1b2c3d"},
		{"load_backend: loaded CUDA backend from /opt/llama.cpp/b11060/libggml-cuda.so\n" +
			"load_backend: loaded CPU backend from /opt/llama.cpp/b11060/libggml-cpu-zen4.so\n" +
			"version: 11060 (a1b2c3d)\nbuilt with cc 13.3.0 for x86_64-linux-gnu\n", "b11060-a1b2c3d+CUDA"},
		{"load_backend: loaded Vulkan backend from x\nload_backend: loaded CUDA backend from y\nversion: 1 (ff)\n", "b1-ff+CUDA+Vulkan"},
		{"error: unknown argument: --version\n", ""},
		{"", ""},
	}
	for _, c := range cases {
		if got := parseVersion(c.out); got != c.want {
			t.Errorf("parseVersion(%q) = %q, want %q", c.out, got, c.want)
		}
	}
}

func TestVersionAsksTheBinaryOnce(t *testing.T) {
	dir := t.TempDir()
	calls := filepath.Join(dir, "calls")
	bin := filepath.Join(dir, "llama-server")
	stub := "#!/bin/sh\necho x >> " + calls + "\necho 'version: 11060 (a1b2c3d)' >&2\n"
	if err := os.WriteFile(bin, []byte(stub), 0o755); err != nil {
		t.Fatal(err)
	}
	rt := New("llamacpp", config.Runtime{Type: config.RuntimeLlamaCpp, Binary: bin}, "")
	if v := rt.Version(); v != "b11060-a1b2c3d" {
		t.Fatalf("Version() = %q", v)
	}
	_ = rt.Version()
	if raw, _ := os.ReadFile(calls); strings.Count(string(raw), "x") != 1 {
		t.Errorf("binary asked %d times, want once", strings.Count(string(raw), "x"))
	}

	missing := New("llamacpp", config.Runtime{Type: config.RuntimeLlamaCpp, Binary: filepath.Join(dir, "nope")}, "")
	if v := missing.Version(); v != "" {
		t.Errorf("missing binary: Version() = %q, want empty", v)
	}
}
