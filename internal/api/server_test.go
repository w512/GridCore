package api

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gridcore/gridcore/internal/config"
	gpufake "github.com/gridcore/gridcore/internal/gpu/fake"
	"github.com/gridcore/gridcore/internal/metrics"
	"github.com/gridcore/gridcore/internal/model"
	"github.com/gridcore/gridcore/internal/runtime"
	rtfake "github.com/gridcore/gridcore/internal/runtime/fake"
	"github.com/gridcore/gridcore/internal/scheduler"
)

var portBase atomic.Int64

func init() { portBase.Store(45000) }

const liveYAML = `
gpu: { device: fake, headroom_mb: 512, poll_interval: 10ms }
runtimes:
  sim: { type: fake, port_range: [%d, %d] }
policy:
  embedding_chunk_size: 4
  interactive_idle_before_background: 10ms
models:
  chat:  { runtime: sim, capabilities: [chat], aliases: [gpt-4o], fake_vram_mb: 9000, fake_load_time: 10ms, parallel: 2 }
  embed: { runtime: sim, capabilities: [embedding], fake_vram_mb: 600, fake_load_time: 5ms, parallel: 4, pinned: true }
  slow:  { runtime: sim, capabilities: [chat], fake_vram_mb: 1000, fake_load_time: 400ms }
`

type live struct {
	ts    *httptest.Server
	rt    *rtfake.Runtime
	sched *scheduler.Scheduler
	cfg   *config.Config
}

func newLive(t *testing.T) *live {
	t.Helper()
	lo := int(portBase.Add(50))
	cfg, err := config.Parse([]byte(fmt.Sprintf(liveYAML, lo, lo+49)))
	if err != nil {
		t.Fatal(err)
	}
	gpu := gpufake.New("FakeGPU", 16000)
	rt := rtfake.New("sim", gpu)
	rt.RequestDelay = 5 * time.Millisecond
	store, _ := model.OpenStore("")
	sched, err := scheduler.New(cfg, map[string]runtime.Runtime{"sim": rt}, gpu, store, metrics.New(), scheduler.Options{
		Tick:            5 * time.Millisecond,
		ShutdownTimeout: 2 * time.Second,
		Logger:          slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = sched.Run(ctx); close(done) }()

	srv := New(cfg, metrics.New(), sched, "test")
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(func() {
		ts.Close()
		cancel()
		<-done
	})
	return &live{ts: ts, rt: rt, sched: sched, cfg: cfg}
}

func (l *live) post(t *testing.T, path, body string, hdr map[string]string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, l.ts.URL+path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func readJSON(t *testing.T, resp *http.Response) map[string]any {
	t.Helper()
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("invalid JSON (%d): %s", resp.StatusCode, raw)
	}
	return out
}

func errCode(t *testing.T, resp *http.Response) string {
	t.Helper()
	out := readJSON(t, resp)
	e, _ := out["error"].(map[string]any)
	code, _ := e["code"].(string)
	return code
}

// ---- static endpoints ----

func TestHealthAndHeaders(t *testing.T) {
	l := newLive(t)
	resp, err := http.Get(l.ts.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 || resp.Header.Get("Server") != "gridcore/test" {
		t.Errorf("status=%d server=%q", resp.StatusCode, resp.Header.Get("Server"))
	}
}

func TestListModelsIncludesAliases(t *testing.T) {
	l := newLive(t)
	resp, err := http.Get(l.ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	out := readJSON(t, resp)
	data := out["data"].([]any)
	ids := map[string]string{}
	for _, d := range data {
		m := d.(map[string]any)
		root, _ := m["root"].(string)
		ids[m["id"].(string)] = root
	}
	if len(ids) != 4 || ids["gpt-4o"] != "chat" || ids["chat"] != "" {
		t.Errorf("models = %+v", ids)
	}
}

func TestMetricsEndpoint(t *testing.T) {
	l := newLive(t)
	resp, err := http.Get(l.ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "go_goroutines") {
		t.Error("metrics output should include go collector series")
	}
}

func TestNoSchedulerAnswers503(t *testing.T) {
	cfg, _ := config.Parse([]byte("runtimes:\n  sim: {type: fake}\nmodels:\n  chat: {runtime: sim, capabilities: [chat], fake_vram_mb: 1}\n"))
	ts := httptest.NewServer(New(cfg, metrics.New(), nil, "test").Handler())
	defer ts.Close()
	resp, _ := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"chat"}`))
	if resp.StatusCode != 503 || errCode(t, resp) != "no_scheduler" {
		t.Errorf("status = %d", resp.StatusCode)
	}
}

// ---- inference ----

func TestChatNonStreaming(t *testing.T) {
	l := newLive(t)
	resp := l.post(t, "/v1/chat/completions", `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, errCode(t, resp))
	}
	if resp.Header.Get(HeaderJobID) == "" || resp.Header.Get(HeaderModel) != "chat" || resp.Header.Get(HeaderQueueMS) == "" {
		t.Errorf("headers = %v", resp.Header)
	}
	out := readJSON(t, resp)
	choices := out["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "ok from chat" {
		t.Errorf("content = %v", msg["content"])
	}
	if l.rt.Instance("chat").Requests() != 1 {
		t.Errorf("requests = %d", l.rt.Instance("chat").Requests())
	}
}

func TestChatStreaming(t *testing.T) {
	l := newLive(t)
	resp := l.post(t, "/v1/chat/completions", `{"model":"chat","stream":true,"messages":[]}`, nil)
	defer resp.Body.Close()
	if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("status=%d ct=%s", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	var events []string
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if line := sc.Text(); strings.HasPrefix(line, "data: ") {
			events = append(events, strings.TrimPrefix(line, "data: "))
		}
	}
	if len(events) != 6 || events[5] != "[DONE]" || !strings.Contains(events[0], "tok0") {
		t.Errorf("events = %v", events)
	}
}

func TestCompletionsOnChatModel(t *testing.T) {
	l := newLive(t)
	resp := l.post(t, "/v1/completions", `{"model":"chat","prompt":"x"}`, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, errCode(t, resp))
	}
	resp.Body.Close()
}

func TestEmbeddingsChunked(t *testing.T) {
	l := newLive(t)
	inputs := make([]string, 10)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("\"doc %d\"", i)
	}
	body := `{"model":"embed","input":[` + strings.Join(inputs, ",") + `]}`
	resp := l.post(t, "/v1/embeddings", body, map[string]string{HeaderClass: "background"})
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, errCode(t, resp))
	}
	if resp.Header.Get(HeaderQueueMS) == "" {
		t.Error("queue header missing")
	}
	out := readJSON(t, resp)
	data := out["data"].([]any)
	if len(data) != 10 {
		t.Fatalf("data = %d items", len(data))
	}
	for i, d := range data {
		item := d.(map[string]any)
		if int(item["index"].(float64)) != i {
			t.Errorf("item %d has index %v", i, item["index"])
		}
		emb := item["embedding"].([]any)
		// fake encodes the chunk-local index in the last dimension: 0..3, 0..3, 0..1
		if int(emb[7].(float64)) != i%4 {
			t.Errorf("item %d embedding marker %v, want %d", i, emb[7], i%4)
		}
	}
	usage := out["usage"].(map[string]any)
	if int(usage["prompt_tokens"].(float64)) != 80 {
		t.Errorf("usage = %v", usage)
	}
	if got := l.rt.Instance("embed").Requests(); got != 3 {
		t.Errorf("upstream requests = %d, want 3 chunks", got)
	}
}

func TestEmbeddingsSingleString(t *testing.T) {
	l := newLive(t)
	resp := l.post(t, "/v1/embeddings", `{"model":"embed","input":"hello"}`, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	out := readJSON(t, resp)
	if len(out["data"].([]any)) != 1 {
		t.Errorf("data = %v", out["data"])
	}
}

func TestRequestErrors(t *testing.T) {
	l := newLive(t)
	cases := []struct {
		path, body string
		hdr        map[string]string
		status     int
		code       string
	}{
		{"/v1/chat/completions", `{"model":"nope"}`, nil, 404, "model_not_found"},
		{"/v1/embeddings", `{"model":"chat","input":"x"}`, nil, 400, "model_capability"},
		{"/v1/chat/completions", `{"model":"chat"}`, map[string]string{HeaderClass: "urgent"}, 400, "invalid_class"},
		{"/v1/chat/completions", `not json`, nil, 400, "invalid_json"},
	}
	for _, c := range cases {
		resp := l.post(t, c.path, c.body, c.hdr)
		if resp.StatusCode != c.status {
			t.Errorf("%s %s: status %d want %d", c.path, c.body, resp.StatusCode, c.status)
		}
		if code := errCode(t, resp); code != c.code {
			t.Errorf("%s %s: code %s want %s", c.path, c.body, code, c.code)
		}
	}
}

func TestQueueTimeoutMapsTo503(t *testing.T) {
	l := newLive(t)
	resp := l.post(t, "/v1/chat/completions", `{"model":"slow","messages":[]}`, map[string]string{HeaderMaxWait: "30"})
	if resp.StatusCode != 503 || resp.Header.Get("Retry-After") == "" {
		t.Fatalf("status=%d retry-after=%q", resp.StatusCode, resp.Header.Get("Retry-After"))
	}
	if code := errCode(t, resp); code != "queue_timeout" {
		t.Errorf("code = %s", code)
	}
}

func TestUpstreamErrorPassthrough(t *testing.T) {
	l := newLive(t)
	l.rt.SetRequestStatus(500)
	resp := l.post(t, "/v1/chat/completions", `{"model":"chat","messages":[]}`, nil)
	if resp.StatusCode != 500 || errCode(t, resp) != "injected" {
		t.Errorf("single-step upstream error not passed through: %d", resp.StatusCode)
	}
	// Chunked: the first failing chunk's response is returned.
	resp = l.post(t, "/v1/embeddings", `{"model":"embed","input":["a","b","c","d","e"]}`, nil)
	if resp.StatusCode != 500 || errCode(t, resp) != "injected" {
		t.Errorf("chunked upstream error not passed through: %d", resp.StatusCode)
	}
	l.rt.SetRequestStatus(0)
	resp = l.post(t, "/v1/chat/completions", `{"model":"chat","messages":[]}`, nil)
	if resp.StatusCode != 200 {
		t.Errorf("recovered request failed: %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestClientDisconnectCancelsJob(t *testing.T) {
	l := newLive(t)
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, l.ts.URL+"/v1/chat/completions", strings.NewReader(`{"model":"slow","messages":[]}`))
	errCh := make(chan error, 1)
	go func() {
		_, err := http.DefaultClient.Do(req)
		errCh <- err
	}()
	time.Sleep(20 * time.Millisecond)
	cancel()
	if err := <-errCh; err == nil {
		t.Fatal("expected client-side cancellation error")
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		st := l.sched.State()
		if len(st.Queued) == 0 && len(st.Running) == 0 {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("job still present after disconnect: %+v", l.sched.State().Queued)
}

func TestAdminEndpoints(t *testing.T) {
	l := newLive(t)
	// unload pinned -> 409
	resp := l.post(t, "/admin/models/embed/unload", "", nil)
	if resp.StatusCode != 409 {
		t.Errorf("unload pinned: %d", resp.StatusCode)
	}
	resp.Body.Close()
	// load chat -> 200, appears resident
	resp = l.post(t, "/admin/models/chat/load", "", nil)
	if resp.StatusCode != 200 {
		t.Fatalf("load: %d", resp.StatusCode)
	}
	resp.Body.Close()
	deadline := time.Now().Add(2 * time.Second)
	for {
		st := l.sched.State()
		ready := false
		for _, r := range st.Resident {
			if r.ID == "chat" && r.State == "ready" {
				ready = true
			}
		}
		if ready || time.Now().After(deadline) {
			if !ready {
				t.Fatal("chat did not become resident")
			}
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	resp = l.post(t, "/admin/models/chat/unload", "", nil)
	if resp.StatusCode != 200 {
		t.Errorf("unload: %d", resp.StatusCode)
	}
	resp.Body.Close()
	resp = l.post(t, "/admin/models/nope/load", "", nil)
	if resp.StatusCode != 404 {
		t.Errorf("unknown: %d", resp.StatusCode)
	}
	resp.Body.Close()
	resp = l.post(t, "/admin/models/chat/enable", "", nil)
	if resp.StatusCode != 200 {
		t.Errorf("enable: %d", resp.StatusCode)
	}
	resp.Body.Close()

	// /admin/state is the scheduler's snapshot
	r2, err := http.Get(l.ts.URL + "/admin/state")
	if err != nil {
		t.Fatal(err)
	}
	st := readJSON(t, r2)
	if _, ok := st["gpu"]; !ok {
		t.Errorf("state = %v", st)
	}
}

func TestConcurrentMixedLoad(t *testing.T) {
	l := newLive(t)
	const n = 12
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			var resp *http.Response
			switch i % 3 {
			case 0:
				resp = l.post(t, "/v1/chat/completions", `{"model":"chat","messages":[]}`, nil)
			case 1:
				resp = l.post(t, "/v1/embeddings", `{"model":"embed","input":["a","b","c","d","e","f"]}`, map[string]string{HeaderClass: "background"})
			default:
				resp = l.post(t, "/v1/chat/completions", `{"model":"chat@batch","stream":true,"messages":[]}`, nil)
			}
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode != 200 {
				errs <- fmt.Errorf("request %d: status %d", i, resp.StatusCode)
				return
			}
			errs <- nil
		}(i)
	}
	for i := 0; i < n; i++ {
		if err := <-errs; err != nil {
			t.Error(err)
		}
	}
	if peak := l.rt.Instance("chat").PeakConcurrency(); peak > 2 {
		t.Errorf("chat parallel=2 but peak concurrency was %d", peak)
	}
	if peak := l.rt.Instance("embed").PeakConcurrency(); peak > 4 {
		t.Errorf("embed parallel=4 but peak concurrency was %d", peak)
	}
}

// TestCancelRaceDoesNotLeakSlots fires many requests that are cancelled at
// roughly the moment they are granted. Whatever the interleaving, every slot
// must be free afterwards.
func TestCancelRaceDoesNotLeakSlots(t *testing.T) {
	l := newLive(t)
	// Warm the model so grants are immediate.
	resp := l.post(t, "/v1/chat/completions", `{"model":"chat","messages":[]}`, nil)
	resp.Body.Close()

	const n = 60
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx, cancel := context.WithCancel(context.Background())
			body := `{"model":"chat","messages":[],"stream":` + fmt.Sprint(i%2 == 0) + `}`
			req, _ := http.NewRequestWithContext(ctx, http.MethodPost, l.ts.URL+"/v1/chat/completions", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			go func() {
				time.Sleep(time.Duration(i%7) * 300 * time.Microsecond)
				cancel()
			}()
			if r, err := http.DefaultClient.Do(req); err == nil {
				io.Copy(io.Discard, r.Body)
				r.Body.Close()
			}
		}(i)
	}
	wg.Wait()

	deadline := time.Now().Add(3 * time.Second)
	for {
		st := l.sched.State()
		busy := 0
		for _, r := range st.Resident {
			busy += r.BusySlots
		}
		if busy == 0 && len(st.Running) == 0 && len(st.Queued) == 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("leaked after cancel race: busy=%d running=%d queued=%d", busy, len(st.Running), len(st.Queued))
		}
		time.Sleep(10 * time.Millisecond)
	}
}
