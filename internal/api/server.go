// Package api is the HTTP surface: OpenAI-compatible endpoints, admin state
// and metrics. It translates requests into jobs and proxies bytes; it holds
// no scheduling policy.
package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"sort"

	"github.com/gridcore/gridcore/internal/config"
	"github.com/gridcore/gridcore/internal/job"
	"github.com/gridcore/gridcore/internal/metrics"
	"github.com/gridcore/gridcore/internal/scheduler"
)

// Scheduler is what the API needs from the control plane.
type Scheduler interface {
	Submit(j *job.Job) (*scheduler.Handle, error)
	State() scheduler.State
	LoadModel(id string) error
	UnloadModel(id string) error
	EnableModel(id string) error
}

// Server wires handlers.
type Server struct {
	cfg     *config.Config
	metrics *metrics.Metrics
	sched   Scheduler
	version string
	mux     *http.ServeMux
	proxy   *proxy

	// MaxBodyBytes bounds request bodies (vision requests carry base64
	// images). Default 32 MB.
	MaxBodyBytes int64
}

// New builds the handler tree. sched may be nil for a control-plane-less
// server (inference endpoints then answer 503).
func New(cfg *config.Config, m *metrics.Metrics, sched Scheduler, version string) *Server {
	s := &Server{
		cfg:          cfg,
		metrics:      m,
		sched:        sched,
		version:      version,
		mux:          http.NewServeMux(),
		proxy:        newProxy(),
		MaxBodyBytes: 32 << 20,
	}
	s.routes()
	return s
}

func (s *Server) routes() {
	s.mux.HandleFunc("GET /health", s.health)
	s.mux.Handle("GET /metrics", s.metrics.Handler())
	s.mux.HandleFunc("GET /admin/state", s.adminState)
	s.mux.HandleFunc("POST /admin/models/{id}/load", s.adminModel("load"))
	s.mux.HandleFunc("POST /admin/models/{id}/unload", s.adminModel("unload"))
	s.mux.HandleFunc("POST /admin/models/{id}/enable", s.adminModel("enable"))
	s.mux.HandleFunc("GET /v1/models", s.listModels)
	s.mux.HandleFunc("POST /v1/chat/completions", s.inference(job.Chat))
	s.mux.HandleFunc("POST /v1/completions", s.inference(job.Completion))
	s.mux.HandleFunc("POST /v1/embeddings", s.inference(job.Embedding))
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
	if s.sched == nil {
		writeJSON(w, http.StatusOK, scheduler.State{Mode: "no scheduler"})
		return
	}
	writeJSON(w, http.StatusOK, s.sched.State())
}

func (s *Server) adminModel(op string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.sched == nil {
			WriteError(w, http.StatusServiceUnavailable, "server_error", "no_scheduler", "scheduler not running")
			return
		}
		id := r.PathValue("id")
		if _, ok := s.cfg.Models[id]; !ok {
			WriteError(w, http.StatusNotFound, "invalid_request_error", "model_not_found", "unknown model "+id)
			return
		}
		var err error
		switch op {
		case "load":
			err = s.sched.LoadModel(id)
		case "unload":
			err = s.sched.UnloadModel(id)
		case "enable":
			err = s.sched.EnableModel(id)
		}
		if err != nil {
			status := http.StatusConflict
			if errors.Is(err, scheduler.ErrShuttingDown) {
				status = http.StatusServiceUnavailable
			}
			WriteError(w, status, "server_error", "admin_"+op+"_failed", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "model": id, "op": op})
	}
}

// listModels answers GET /v1/models in OpenAI shape, including aliases as
// separate entries so `client.models.list()` shows every name that works.
func (s *Server) listModels(w http.ResponseWriter, _ *http.Request) {
	type entry struct {
		ID           string   `json:"id"`
		Object       string   `json:"object"`
		OwnedBy      string   `json:"owned_by"`
		Root         string   `json:"root,omitempty"`
		Capabilities []string `json:"capabilities,omitempty"`
	}
	var data []entry
	for id, m := range s.cfg.Models {
		data = append(data, entry{ID: id, Object: "model", OwnedBy: "gridcore", Capabilities: m.Capabilities})
		for _, a := range m.Aliases {
			data = append(data, entry{ID: a, Object: "model", OwnedBy: "gridcore", Root: id})
		}
	}
	sort.Slice(data, func(i, j int) bool { return data[i].ID < data[j].ID })
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// WriteError emits an OpenAI-shaped error body.
func WriteError(w http.ResponseWriter, status int, typ, code, msg string) {
	writeJSON(w, status, errorBody(typ, code, msg))
}

func errorBody(typ, code, msg string) map[string]any {
	return map[string]any{
		"error": map[string]any{"message": msg, "type": typ, "code": code, "param": nil},
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func newJobID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
