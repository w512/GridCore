package api

import (
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"

	"github.com/w512/gridcore/internal/config"
)

func allowedHosts(cfg config.Server) map[string]bool {
	hosts := map[string]bool{"localhost": true}
	if host, _, err := net.SplitHostPort(cfg.Listen); err == nil && host != "" && host != "*" {
		ip, err := netip.ParseAddr(host)
		if err != nil || !ip.Unmap().IsUnspecified() {
			hosts[canonicalHost(host)] = true
		}
	}
	for _, host := range cfg.AllowedHosts {
		hosts[canonicalHost(host)] = true
	}
	return hosts
}

func canonicalHost(host string) string {
	if ip, err := netip.ParseAddr(host); err == nil {
		return ip.Unmap().String()
	}
	return strings.ToLower(strings.TrimSuffix(host, "."))
}

func (s *Server) hostAllowed(authority string) bool {
	u, err := url.Parse("http://" + authority)
	if err != nil || u.Host != authority || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || u.Hostname() == "" {
		return false
	}
	host := canonicalHost(u.Hostname())
	if ip, err := netip.ParseAddr(host); err == nil && ip.IsLoopback() {
		return true
	}
	// Compare literal names only. Resolving an untrusted hostname to a
	// loopback IP here would reintroduce DNS rebinding.
	return s.allowedHosts[host]
}

func (s *Server) protectBrowserRequests(next http.Handler) http.Handler {
	csrf := http.NewCrossOriginProtection()
	csrf.SetDenyHandler(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		WriteError(w, http.StatusForbidden, "invalid_request_error", "cross_origin_request", "cross-origin requests are not allowed")
	}))
	protected := csrf.Handler(next)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// This also guards GET endpoints: origin checks alone cannot stop
		// rebinding requests whose Origin and Host both name the attacker.
		if !s.hostAllowed(r.Host) {
			WriteError(w, http.StatusForbidden, "invalid_request_error", "invalid_host", "request Host is not allowed")
			return
		}
		protected.ServeHTTP(w, r)
	})
}
