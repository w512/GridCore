// Package procname finds the executable name of a process by PID, for
// showing who else is using the GPU.
//
// On Linux the name comes from /proc/<pid>/comm, on macOS from ps. Lookups
// on macOS spawn a process, so callers that poll keep the names in a Cache
// and ask the OS only about PIDs they have not seen yet.
package procname

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// psTimeout bounds one ps call; a name is not worth stalling a GPU poll for.
const psTimeout = 2 * time.Second

// Name returns the executable name of pid, or "" if it can't be determined.
func Name(pid int) string {
	if pid <= 0 {
		return ""
	}
	switch runtime.GOOS {
	case "linux":
		return FromProc("/proc", pid)
	case "darwin":
		return fromPS(pid)
	}
	return ""
}

// FromProc reads <procRoot>/<pid>/comm (Linux layout). Exported so tests can use a fake tree.
func FromProc(procRoot string, pid int) string {
	if pid <= 0 {
		return ""
	}
	b, err := os.ReadFile(filepath.Join(procRoot, strconv.Itoa(pid), "comm"))
	if err != nil {
		return ""
	}
	return strings.TrimRight(string(b), "\r\n")
}

// fromPS asks ps for the process's command name. On macOS that is the path
// of the executable, without arguments.
func fromPS(pid int) string {
	ctx, cancel := context.WithTimeout(context.Background(), psTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, "ps", "-o", "comm=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return "" // no such process
	}
	return parsePS(out)
}

// parsePS turns `ps -o comm=` output into a base name.
func parsePS(out []byte) string {
	line, _, _ := bytes.Cut(out, []byte("\n"))
	s := strings.TrimSpace(string(line))
	if s == "" {
		return ""
	}
	return filepath.Base(s)
}

// Cache remembers process names by PID. Safe for concurrent use.
type Cache struct {
	lookup func(pid int) string

	mu    sync.Mutex
	names map[int]string
}

// NewCache returns a cache that resolves names with lookup (Name if nil).
func NewCache(lookup func(pid int) string) *Cache {
	if lookup == nil {
		lookup = Name
	}
	return &Cache{lookup: lookup, names: map[int]string{}}
}

// Names returns the names of pids. Only PIDs missing from the previous call
// are looked up; the others keep the name they had. PIDs not in pids are
// forgotten, so a PID that comes back later (possibly reused by another
// process) is looked up again. A name that can't be determined is "" and is
// not retried while the PID stays.
func (c *Cache) Names(pids []int) map[int]string {
	c.mu.Lock()
	defer c.mu.Unlock()
	next := make(map[int]string, len(pids))
	for _, pid := range pids {
		if _, ok := next[pid]; ok {
			continue
		}
		name, ok := c.names[pid]
		if !ok {
			name = c.lookup(pid)
		}
		next[pid] = name
	}
	c.names = next
	out := make(map[int]string, len(next))
	for pid, name := range next {
		out[pid] = name
	}
	return out
}
