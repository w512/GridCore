package api

import (
	_ "embed"
	"net/http"
)

//go:embed ui/index.html
var dashboardHTML []byte

// dashboardCSP keeps the page self-contained: its own inline script and
// styles, requests to this daemon only, nothing loaded from elsewhere, and
// no framing.
const dashboardCSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; " +
	"connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"

// adminUI serves the dashboard: one page compiled into the binary that polls
// /admin/state and keeps a few minutes of history for its charts in the
// browser. Read-only, like `gridcore status`.
func (s *Server) adminUI(w http.ResponseWriter, _ *http.Request) {
	h := w.Header()
	h.Set("Content-Type", "text/html; charset=utf-8")
	h.Set("Content-Security-Policy", dashboardCSP)
	h.Set("X-Content-Type-Options", "nosniff")
	h.Set("Cache-Control", "no-cache")
	_, _ = w.Write(dashboardHTML)
}

// root sends a browser opened on the daemon's address to the dashboard.
func (s *Server) root(w http.ResponseWriter, r *http.Request) {
	http.Redirect(w, r, "/admin/ui", http.StatusFound)
}
