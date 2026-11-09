package api

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/w512/gridcore/internal/config"
	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/metrics"
	"github.com/w512/gridcore/internal/scheduler"
)

type securityScheduler struct{ calls []string }

func (s *securityScheduler) Submit(*job.Job) (*scheduler.Handle, error) {
	s.calls = append(s.calls, "submit")
	return nil, errors.New("test stops before dispatch")
}
func (s *securityScheduler) State() scheduler.State {
	s.calls = append(s.calls, "state")
	return scheduler.State{Mode: "test"}
}
func (s *securityScheduler) LoadModel(string) error {
	s.calls = append(s.calls, "load")
	return nil
}
func (s *securityScheduler) UnloadModel(string) error {
	s.calls = append(s.calls, "unload")
	return nil
}
func (s *securityScheduler) EnableModel(string) error {
	s.calls = append(s.calls, "enable")
	return nil
}

func securityHandler(t *testing.T, server config.Server) (http.Handler, *securityScheduler) {
	t.Helper()
	cfg := parseCfg(t)
	cfg.Server = server
	spy := &securityScheduler{}
	return New(cfg, metrics.New(), spy, "test").Handler(), spy
}

func TestRejectCrossOriginBeforeScheduling(t *testing.T) {
	for _, path := range []string{
		"/admin/models/chat/load", "/admin/models/chat/unload", "/admin/models/chat/enable",
		"/v1/chat/completions", "/v1/completions", "/v1/embeddings",
	} {
		t.Run(path, func(t *testing.T) {
			handler, spy := securityHandler(t, config.Server{Listen: "127.0.0.1:8080"})
			r := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080"+path, strings.NewReader(`{"model":"chat"}`))
			r.Header.Set("Origin", "https://untrusted.example")
			r.Header.Set("Sec-Fetch-Site", "cross-site")
			r.Header.Set("Content-Type", "text/plain")
			if strings.HasPrefix(path, "/admin/") {
				r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
			}
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, r)
			if w.Code != http.StatusForbidden || errCode(t, w.Result()) != "cross_origin_request" {
				t.Fatalf("status=%d, want cross-origin rejection", w.Code)
			}
			if len(spy.calls) != 0 {
				t.Fatalf("cross-origin request reached scheduler: %v", spy.calls)
			}
		})
	}
}

func TestBrowserOriginAndCLICompatibility(t *testing.T) {
	for _, tc := range []struct {
		name, origin, site string
		allowed            bool
	}{
		{"CLI without browser headers", "", "", true},
		{"same origin with fetch metadata", "http://127.0.0.1:8080", "same-origin", true},
		{"same origin without fetch metadata", "http://127.0.0.1:8080", "", true},
		{"foreign origin without fetch metadata", "https://untrusted.example", "", false},
		{"opaque origin", "null", "", false},
		{"malformed origin", "://invalid", "", false},
		{"cross-site without origin", "", "cross-site", false},
		{"same-site different origin", "http://localhost:8080", "same-site", false},
		{"different port", "http://127.0.0.1:8081", "", false},
		{"allowed host is not a trusted origin", "https://proxy.example", "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handler, spy := securityHandler(t, config.Server{Listen: "127.0.0.1:8080", AllowedHosts: []string{"proxy.example"}})
			r := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080/admin/models/chat/load", nil)
			if tc.origin != "" {
				r.Header.Set("Origin", tc.origin)
			}
			if tc.site != "" {
				r.Header.Set("Sec-Fetch-Site", tc.site)
			}
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, r)
			if tc.allowed {
				if w.Code != http.StatusOK || len(spy.calls) != 1 || spy.calls[0] != "load" {
					t.Fatalf("status=%d calls=%v", w.Code, spy.calls)
				}
			} else if w.Code != http.StatusForbidden || len(spy.calls) != 0 {
				t.Fatalf("status=%d calls=%v, want rejection before scheduler", w.Code, spy.calls)
			}
		})
	}
}

func TestRejectRebindingHostOnEveryEndpoint(t *testing.T) {
	for _, tc := range []struct{ method, path string }{
		{"GET", "/"}, {"GET", "/health"}, {"GET", "/metrics"},
		{"GET", "/admin/ui"}, {"GET", "/admin/state"}, {"GET", "/v1/models"},
		{"HEAD", "/admin/state"}, {"OPTIONS", "/v1/chat/completions"},
		{"POST", "/admin/models/chat/load"}, {"POST", "/admin/models/chat/unload"},
		{"POST", "/admin/models/chat/enable"}, {"POST", "/v1/chat/completions"},
		{"POST", "/v1/completions"}, {"POST", "/v1/embeddings"},
	} {
		t.Run(tc.method+tc.path, func(t *testing.T) {
			handler, spy := securityHandler(t, config.Server{Listen: "127.0.0.1:8080"})
			r := httptest.NewRequest(tc.method, "http://rebound.example:8080"+tc.path, nil)
			// Rebinding is same-origin to the browser, so a CSRF check alone
			// would accept it. Forwarded headers must not bypass Host checks.
			r.Header.Set("Origin", "http://rebound.example:8080")
			r.Header.Set("Sec-Fetch-Site", "same-origin")
			r.Header.Set("X-Forwarded-Host", "127.0.0.1:8080")
			r.Header.Set("Forwarded", "host=localhost:8080")
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, r)
			if w.Code != http.StatusForbidden || errCode(t, w.Result()) != "invalid_host" {
				t.Fatalf("status=%d, want Host rejection", w.Code)
			}
			if len(spy.calls) != 0 {
				t.Fatalf("rebinding request reached scheduler: %v", spy.calls)
			}
		})
	}
}

func TestAllowedRequestHosts(t *testing.T) {
	for _, tc := range []struct {
		name, listen, host string
		extra              []string
		allowed            bool
	}{
		{"loopback", "127.0.0.1:8080", "127.0.0.1:8080", nil, true},
		{"loopback subnet", "127.0.0.1:8080", "127.0.0.2:8080", nil, true},
		{"IPv6 loopback", "127.0.0.1:8080", "[::1]:8080", nil, true},
		{"mapped loopback", "127.0.0.1:8080", "[::ffff:127.0.0.1]:8080", nil, true},
		{"localhost", "127.0.0.1:8080", "localhost:8080", nil, true},
		{"case and trailing dot", "127.0.0.1:8080", "LOCALHOST.:8080", nil, true},
		{"without port", "127.0.0.1:8080", "localhost", nil, true},
		{"concrete LAN IP", "192.168.1.5:8080", "192.168.1.5:8080", nil, true},
		{"concrete hostname", "gridcore.lan:8080", "GRIDCORE.LAN.:8080", nil, true},
		{"concrete IPv6", "[fd00::5]:8080", "[fd00:0::5]:8080", nil, true},
		{"explicit proxy hostname", "127.0.0.1:8080", "proxy.example:443", []string{"Proxy.Example"}, true},
		{"wildcard IPv4 is not a Host wildcard", "0.0.0.0:8080", "rebound.example:8080", nil, false},
		{"wildcard IPv6 is not a Host wildcard", "[::]:8080", "rebound.example:8080", nil, false},
		{"empty listen host is not a Host wildcard", ":8080", "rebound.example:8080", nil, false},
		{"wildcard listen with explicit LAN IP", "0.0.0.0:8080", "192.168.1.5:8080", []string{"192.168.1.5"}, true},
		{"wildcard IPv4 literal", "0.0.0.0:8080", "0.0.0.0:8080", nil, false},
		{"wildcard IPv6 literal", "[::]:8080", "[::]:8080", nil, false},
		{"loopback DNS suffix", "127.0.0.1:8080", "127.0.0.1.untrusted.example:8080", nil, false},
		{"localhost DNS suffix", "127.0.0.1:8080", "localhost.untrusted.example:8080", nil, false},
		{"empty host", "127.0.0.1:8080", "", nil, false},
		{"userinfo", "127.0.0.1:8080", "untrusted.example@localhost:8080", nil, false},
		{"path", "127.0.0.1:8080", "localhost:8080/untrusted", nil, false},
		{"invalid port", "127.0.0.1:8080", "localhost:invalid", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handler, _ := securityHandler(t, config.Server{Listen: tc.listen, AllowedHosts: tc.extra})
			r := httptest.NewRequest(http.MethodGet, "http://127.0.0.1:8080/health", nil)
			r.Host = tc.host
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, r)
			want := http.StatusForbidden
			if tc.allowed {
				want = http.StatusOK
			}
			if w.Code != want {
				t.Fatalf("Host %q: status=%d want=%d", tc.host, w.Code, want)
			}
		})
	}
}

func TestInferenceRequiresJSONBeforeScheduling(t *testing.T) {
	for _, endpoint := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"model":"chat","messages":[]}`},
		{"/v1/completions", `{"model":"raw","prompt":"hi"}`},
		{"/v1/embeddings", `{"model":"embed","input":"hi"}`},
	} {
		for _, contentType := range []string{"", "text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=test", "application/json; charset", "application/json, text/plain"} {
			t.Run(endpoint.path+"/"+contentType, func(t *testing.T) {
				handler, spy := securityHandler(t, config.Server{Listen: "127.0.0.1:8080"})
				r := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080"+endpoint.path, strings.NewReader(endpoint.body))
				if contentType != "" {
					r.Header.Set("Content-Type", contentType)
				}
				w := httptest.NewRecorder()
				handler.ServeHTTP(w, r)
				if w.Code != http.StatusUnsupportedMediaType || errCode(t, w.Result()) != "unsupported_media_type" {
					t.Fatalf("status=%d, want media type rejection", w.Code)
				}
				if len(spy.calls) != 0 {
					t.Fatalf("non-JSON request reached scheduler: %v", spy.calls)
				}
			})
		}
	}
}
