package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gridcore/gridcore/internal/config"
	"github.com/gridcore/gridcore/internal/metrics"
)

func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	cfg, err := config.Parse([]byte(`
runtimes:
  sim: {type: fake}
models:
  chat: {runtime: sim, capabilities: [chat], aliases: [gpt-4o]}
  embed: {runtime: sim, capabilities: [embedding]}
`))
	if err != nil {
		t.Fatal(err)
	}
	srv := New(cfg, metrics.New(), nil, "test")
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts
}

func TestHealthAndHeaders(t *testing.T) {
	ts := newTestServer(t)
	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 || resp.Header.Get("Server") != "gridcore/test" {
		t.Errorf("status=%d server=%q", resp.StatusCode, resp.Header.Get("Server"))
	}
}

func TestListModelsIncludesAliases(t *testing.T) {
	ts := newTestServer(t)
	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Data []struct {
			ID   string `json:"id"`
			Root string `json:"root"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	ids := map[string]string{}
	for _, d := range out.Data {
		ids[d.ID] = d.Root
	}
	if len(ids) != 3 || ids["gpt-4o"] != "chat" || ids["chat"] != "" {
		t.Errorf("models = %+v", ids)
	}
}

func TestMetricsEndpoint(t *testing.T) {
	ts := newTestServer(t)
	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), "go_goroutines") {
		t.Error("metrics output should include go collector series")
	}
}

func TestInferenceNotImplementedYet(t *testing.T) {
	ts := newTestServer(t)
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotImplemented {
		t.Errorf("status = %d", resp.StatusCode)
	}
	var out struct {
		Error struct{ Code string } `json:"error"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&out)
	if out.Error.Code != "not_implemented" {
		t.Errorf("error = %+v", out)
	}
}
