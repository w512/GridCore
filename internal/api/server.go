// Package api is the HTTP surface: OpenAI-compatible endpoints, admin state
// and metrics. It translates requests into jobs and proxies bytes; it holds
// no scheduling policy.
package api

import (
	"encoding/json"
	"net/http"
	"sort"
	"time"

	"github.com/gridcore/gridcore/internal/config"
	"github.com/gridcore/gridcore/internal/metrics"
)

// State is the JSON document served at /admin/state. The scheduler fills it;
// the API only serialises it.
type State struct {
	Now      time.Time       `json:"now"`
	GPU      GPUState        `json:"gpu"`
	Resident []ResidentModel `json:"resident"`
	Running  []JobState      `json:"running"`
	Queued   []JobState      `json:"queued"`
	Events   []Event         `json:"recent_events"`
}

type GPUState struct {
	Name       string `json:"name"`
	TotalMB    int    `json:"total_mb"`
	BudgetMB   int    `json:"budget_mb"` // total or configured limit, minus headroom
	UsedMB     int    `json:"used_mb"`
	ReservedMB int    `json:"reserved_mb"`
	UtilPct    int    `json:"util_pct"`
	Mode       string `json:"mode"` // interactive | background | idle
}

type ResidentModel struct {
	ID          string    `json:"id"`
	Tier        string    `json:"tier"` // pinned | hot | cold
	VRAMMB      int       `json:"vram_mb"`
	Slots       int       `json:"slots"`
	BusySlots   int       `json:"busy_slots"`
	LastUsed    time.Time `json:"last_used"`
	LoadedAt    time.Time `json:"loaded_at"`
	PID         int       `json:"pid"`
	Addr        string    `json:"addr"`
	State       string    `json:"state"` // loading | ready | draining
	LoadedInSec float64   `json:"loaded_in_sec"`
}

type JobState struct {
	ID       string        `json:"id"`
	Class    string        `json:"class"`
	Kind     string        `json:"kind"`
	Model    string        `json:"model"`
	State    string        `json:"state"`
	Waited   time.Duration `json:"waited_ms"`
	Step     int           `json:"step"`
	Steps    int           `json:"steps"`
	Enqueued time.Time     `json:"enqueued"`
}

type Event struct {
	At      time.Time `json:"at"`
	Kind    string    `json:"kind"` // enqueue|dispatch|complete|load|evict|crash|timeout|cancel
	Subject string    `json:"subject"`
	Detail  string    `json:"detail,omitempty"`
}

// StateProvider is implemented by the scheduler.
type StateProvider interface {
	State() State
}

// Server wires handlers. Inference endpoints are attached by the proxy in
// M1; until then they answer 501 so clients get a clear signal.
type Server struct {
	cfg     *config.Config
	metrics *metrics.Metrics
	state   StateProvider
	version string
	mux     *http.ServeMux
}

// New builds the handler tree.
func New(cfg *config.Config, m *metrics.Metrics, sp StateProvider, version string) *Server {
	s := &Server{cfg: cfg, metrics: m, state: sp, version: version, mux: http.NewServeMux()}
	s.routes()
	return s
}

func (s *Server) routes() {
	s.mux.HandleFunc("GET /health", s.health)
	s.mux.Handle("GET /metrics", s.metrics.Handler())
	s.mux.HandleFunc("GET /admin/state", s.adminState)
	s.mux.HandleFunc("GET /v1/models", s.listModels)

	for _, p := range []string{"/v1/chat/completions", "/v1/completions", "/v1/embeddings"} {
		s.mux.HandleFunc("POST "+p, s.notImplemented)
	}
}

// Handler returns the root handler with common response headers applied.
func (s *Server) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Server", "gridcore/"+s.version)
		s.mux.ServeHTTP(w, r)
	})
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "version": s.version})
}

func (s *Server) adminState(w http.ResponseWriter, _ *http.Request) {
	if s.state == nil {
		writeJSON(w, http.StatusOK, State{Now: time.Now()})
		return
	}
	writeJSON(w, http.StatusOK, s.state.State())
}

// listModels answers GET /v1/models in OpenAI shape, including aliases as
// separate entries so `client.models.list()` shows every name that works.
func (s *Server) listModels(w http.ResponseWriter, _ *http.Request) {
	type entry struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
		Root    string `json:"root,omitempty"`
	}
	var data []entry
	for id, m := range s.cfg.Models {
		data = append(data, entry{ID: id, Object: "model", OwnedBy: "gridcore"})
		for _, a := range m.Aliases {
			data = append(data, entry{ID: a, Object: "model", OwnedBy: "gridcore", Root: id})
		}
	}
	sort.Slice(data, func(i, j int) bool { return data[i].ID < data[j].ID })
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (s *Server) notImplemented(w http.ResponseWriter, r *http.Request) {
	WriteError(w, http.StatusNotImplemented, "not_implemented", "not_implemented",
		"GridCore: "+r.URL.Path+" is not wired to the scheduler yet (milestone M1)")
}

// WriteError emits an OpenAI-shaped error body.
func WriteError(w http.ResponseWriter, status int, typ, code, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{"message": msg, "type": typ, "code": code, "param": nil},
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}
