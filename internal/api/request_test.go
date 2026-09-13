package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/w512/gridcore/internal/config"
	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/scheduler"
)

func parseCfg(t *testing.T) *config.Config {
	t.Helper()
	cfg, err := config.Parse([]byte(`
runtimes:
  sim: {type: fake}
models:
  chat:  {runtime: sim, capabilities: [chat], aliases: [gpt-4o], fake_vram_mb: 9000}
  raw:   {runtime: sim, capabilities: [completion], fake_vram_mb: 1000}
  embed: {runtime: sim, capabilities: [embedding], fake_vram_mb: 600}
policy:
  embedding_chunk_size: 4
`))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func parse(t *testing.T, cfg *config.Config, kind job.Kind, body string, hdr map[string]string) (*inferenceRequest, *apiError) {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/v1/x", strings.NewReader(body))
	for k, v := range hdr {
		r.Header.Set(k, v)
	}
	return parseInferenceRequest(r, kind, cfg, 1<<20)
}

func TestParseClassPrecedence(t *testing.T) {
	cfg := parseCfg(t)
	cases := []struct {
		name string
		body string
		hdr  map[string]string
		want job.Class
	}{
		{"default", `{"model":"chat"}`, nil, job.Interactive},
		{"suffix", `{"model":"chat@batch"}`, nil, job.Batch},
		{"body beats suffix", `{"model":"chat@batch","gridcore":{"class":"background"}}`, nil, job.Background},
		{"header beats body", `{"model":"chat","gridcore":{"class":"background"}}`, map[string]string{HeaderClass: "batch"}, job.Batch},
		{"header case-insensitive", `{"model":"chat"}`, map[string]string{HeaderClass: " Background "}, job.Background},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req, aerr := parse(t, cfg, job.Chat, c.body, c.hdr)
			if aerr != nil {
				t.Fatalf("unexpected error: %v", aerr.msg)
			}
			if req.class != c.want {
				t.Errorf("class = %s want %s", req.class, c.want)
			}
		})
	}
}

func TestParseStripsGridCoreAndResolvesAlias(t *testing.T) {
	cfg := parseCfg(t)
	req, aerr := parse(t, cfg, job.Chat, `{"model":"gpt-4o@background","gridcore":{"class":"batch","max_wait_ms":250},"messages":[]}`, nil)
	if aerr != nil {
		t.Fatal(aerr.msg)
	}
	if req.modelID != "chat" || req.class != job.Batch || req.maxWait != 250*time.Millisecond {
		t.Errorf("req = %+v", req)
	}
	var out map[string]any
	if err := json.Unmarshal(req.marshalBody(), &out); err != nil {
		t.Fatal(err)
	}
	if _, has := out["gridcore"]; has {
		t.Error("gridcore field must be stripped before proxying")
	}
	if out["model"] != "chat" {
		t.Errorf("model must be rewritten to the resolved id, got %v", out["model"])
	}
	if _, has := out["messages"]; !has {
		t.Error("other fields must be preserved")
	}
}

func TestParseMaxWaitHeaderBeatsBody(t *testing.T) {
	cfg := parseCfg(t)
	req, aerr := parse(t, cfg, job.Chat, `{"model":"chat","gridcore":{"max_wait_ms":250}}`, map[string]string{HeaderMaxWait: "1000"})
	if aerr != nil {
		t.Fatal(aerr.msg)
	}
	if req.maxWait != time.Second {
		t.Errorf("max_wait = %v", req.maxWait)
	}
}

func TestParseErrors(t *testing.T) {
	cfg := parseCfg(t)
	cases := []struct {
		name   string
		kind   job.Kind
		body   string
		hdr    map[string]string
		status int
		code   string
	}{
		{"not json", job.Chat, `nope`, nil, 400, "invalid_json"},
		{"no model", job.Chat, `{"messages":[]}`, nil, 400, "model_required"},
		{"model not string", job.Chat, `{"model":5}`, nil, 400, "invalid_model"},
		{"unknown model", job.Chat, `{"model":"llama"}`, nil, 404, "model_not_found"},
		{"bad class suffix", job.Chat, `{"model":"chat@urgent"}`, nil, 400, "invalid_class"},
		{"bad class header", job.Chat, `{"model":"chat"}`, map[string]string{HeaderClass: "now"}, 400, "invalid_class"},
		{"bad gridcore", job.Chat, `{"model":"chat","gridcore":"x"}`, nil, 400, "invalid_gridcore"},
		{"bad max wait", job.Chat, `{"model":"chat"}`, map[string]string{HeaderMaxWait: "-1"}, 400, "invalid_max_wait"},
		{"embeddings on chat model", job.Embedding, `{"model":"chat","input":"x"}`, nil, 400, "model_capability"},
		{"chat on embed model", job.Chat, `{"model":"embed"}`, nil, 400, "model_capability"},
		{"chat on completion-only model", job.Chat, `{"model":"raw"}`, nil, 400, "model_capability"},
		{"embeddings without input", job.Embedding, `{"model":"embed"}`, nil, 400, "input_required"},
		{"embeddings empty input", job.Embedding, `{"model":"embed","input":[]}`, nil, 400, "input_empty"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, aerr := parse(t, cfg, c.kind, c.body, c.hdr)
			if aerr == nil {
				t.Fatal("expected error")
			}
			if aerr.status != c.status || aerr.code != c.code {
				t.Errorf("got %d %s, want %d %s", aerr.status, aerr.code, c.status, c.code)
			}
		})
	}
}

func TestParseCompletionsAcceptsChatModels(t *testing.T) {
	cfg := parseCfg(t)
	for _, m := range []string{"chat", "raw"} {
		if _, aerr := parse(t, cfg, job.Completion, `{"model":"`+m+`","prompt":"x"}`, nil); aerr != nil {
			t.Errorf("%s: %s", m, aerr.msg)
		}
	}
}

func TestParseEmbeddingsChunking(t *testing.T) {
	cfg := parseCfg(t)
	req, aerr := parse(t, cfg, job.Embedding, `{"model":"embed","input":["a","b","c","d","e","f","g","h","i","j"]}`, nil)
	if aerr != nil {
		t.Fatal(aerr.msg)
	}
	if req.steps != 3 || len(req.input) != 10 {
		t.Errorf("steps=%d input=%d", req.steps, len(req.input))
	}
	var chunk struct {
		Input []string `json:"input"`
		Model string   `json:"model"`
	}
	if err := json.Unmarshal(req.chunkBody(8, 10), &chunk); err != nil {
		t.Fatal(err)
	}
	if len(chunk.Input) != 2 || chunk.Input[0] != "i" || chunk.Model != "embed" {
		t.Errorf("chunk = %+v", chunk)
	}

	req, _ = parse(t, cfg, job.Embedding, `{"model":"embed","input":["a","b"]}`, nil)
	if req.steps != 1 {
		t.Errorf("small array should be one step, got %d", req.steps)
	}
	req, _ = parse(t, cfg, job.Embedding, `{"model":"embed","input":"hello"}`, nil)
	if req.steps != 1 || req.input != nil {
		t.Errorf("string input should be one step, got %+v", req)
	}
}

func TestParseBodyTooLarge(t *testing.T) {
	cfg := parseCfg(t)
	r := httptest.NewRequest(http.MethodPost, "/v1/x", strings.NewReader(`{"model":"chat","messages":"`+strings.Repeat("x", 100)+`"}`))
	_, aerr := parseInferenceRequest(r, job.Chat, cfg, 50)
	if aerr == nil || aerr.status != http.StatusRequestEntityTooLarge {
		t.Fatalf("want 413, got %+v", aerr)
	}
}

func TestExtractUsage(t *testing.T) {
	u := extractUsage([]byte(`{"usage":{"prompt_tokens":12,"completion_tokens":34},"timings":{"prompt_per_second":150.5,"predicted_per_second":22.1}}`))
	if u.PromptTokens != 12 || u.CompletionTokens != 34 || u.PromptTPS != 150.5 || u.GenTPS != 22.1 {
		t.Errorf("usage = %+v", u)
	}
	if u := extractUsage([]byte(`garbage`)); u != (scheduler.Usage{}) {
		t.Errorf("garbage should yield zero usage, got %+v", u)
	}
}

func TestSSETailFindsFinalChunkAcrossReads(t *testing.T) {
	var tl sseTail
	// Split awkwardly across reads, CRLF line endings, a non-final chunk with
	// no timings, then the llama-server-style final chunk, then [DONE].
	parts := []string{
		"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\r\n\r\ndata: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"tim",
		"ings\":{\"prompt_per_second\":200.5,\"predicted_per_second\":31.25},\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":17}}\r\n",
		"\r\ndata: [DONE]\n\n",
	}
	for _, p := range parts {
		tl.observe([]byte(p))
	}
	u := tl.usage()
	if u.PromptTokens != 9 || u.CompletionTokens != 17 || u.PromptTPS != 200.5 || u.GenTPS != 31.25 {
		t.Errorf("usage = %+v", u)
	}
	var empty sseTail
	empty.observe([]byte("data: {\"choices\":[]}\n\ndata: [DONE]\n\n"))
	if empty.usage() != (scheduler.Usage{}) {
		t.Error("stream without timings should yield zero usage")
	}
}
