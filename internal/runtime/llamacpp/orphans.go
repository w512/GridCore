package llamacpp

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Registry records live child processes on disk so that a GridCore restart
// (or crash) can reap instances left over from the previous run instead of
// leaking GPU memory to zombie llama-servers.
type Registry struct {
	path string
	mu   sync.Mutex
	live map[int]RegistryEntry
}

// RegistryEntry is one recorded child.
type RegistryEntry struct {
	PID       int       `json:"pid"`
	Model     string    `json:"model"`
	Port      int       `json:"port"`
	Binary    string    `json:"binary"`
	StartedAt time.Time `json:"started_at"`
}

// OpenRegistry loads path (missing = empty).
func OpenRegistry(path string) (*Registry, error) {
	r := &Registry{path: path, live: map[int]RegistryEntry{}}
	raw, err := os.ReadFile(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		return r, nil
	case err != nil:
		return nil, err
	}
	var list []RegistryEntry
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("instances registry %s: %w", path, err)
	}
	for _, e := range list {
		r.live[e.PID] = e
	}
	return r, nil
}

// ReapOrphans kills every recorded process that is still alive and looks
// like one of ours. It returns the entries it acted on and clears the
// registry.
//
// PID reuse is guarded by inspecting the process command line when the OS
// exposes it: a llama-server we started has our binary path and its port
// in argv. /proc/<pid>/exe is deliberately not used: for a script (or a
// wrapper) it points at the interpreter, which would make us skip and leak
// the very process we are trying to reap.
func (r *Registry) ReapOrphans() []RegistryEntry {
	r.mu.Lock()
	defer r.mu.Unlock()
	var reaped []RegistryEntry
	for pid, e := range r.live {
		if !alive(pid) {
			continue
		}
		if cmdline, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid)); err == nil && len(cmdline) > 0 {
			argv := strings.Split(strings.TrimRight(string(cmdline), "\x00"), "\x00")
			if !looksLikeOurs(argv, e) {
				continue // PID was reused by something else
			}
		}
		_ = syscall.Kill(-pid, syscall.SIGTERM)
		_ = syscall.Kill(pid, syscall.SIGTERM)
		reaped = append(reaped, e)
	}
	if len(reaped) > 0 {
		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) {
			any := false
			for _, e := range reaped {
				if alive(e.PID) {
					any = true
				}
			}
			if !any {
				break
			}
			time.Sleep(100 * time.Millisecond)
		}
		for _, e := range reaped {
			if alive(e.PID) {
				_ = syscall.Kill(-e.PID, syscall.SIGKILL)
				_ = syscall.Kill(e.PID, syscall.SIGKILL)
			}
		}
	}
	r.live = map[int]RegistryEntry{}
	_ = r.saveLocked()
	return reaped
}

// Add records a child.
func (r *Registry) Add(e RegistryEntry) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.live[e.PID] = e
	_ = r.saveLocked()
}

// Remove forgets a child.
func (r *Registry) Remove(pid int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.live, pid)
	_ = r.saveLocked()
}

func (r *Registry) saveLocked() error {
	if r.path == "" {
		return nil
	}
	list := make([]RegistryEntry, 0, len(r.live))
	for _, e := range r.live {
		list = append(list, e)
	}
	raw, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(r.path), 0o755); err != nil {
		return err
	}
	tmp := r.path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, r.path)
}

// looksLikeOurs reports whether argv is the llama-server we recorded: it
// mentions the binary (by path or base name, for scripts run through an
// interpreter) and listens on the recorded port.
func looksLikeOurs(argv []string, e RegistryEntry) bool {
	hasBinary, hasPort := false, false
	base := filepath.Base(e.Binary)
	port := strconv.Itoa(e.Port)
	for i, a := range argv {
		if a == e.Binary || filepath.Base(a) == base {
			hasBinary = true
		}
		if (a == "--port" && i+1 < len(argv) && argv[i+1] == port) || a == "--port="+port {
			hasPort = true
		}
	}
	return hasBinary && hasPort
}

func alive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}
