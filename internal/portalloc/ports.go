// Package portalloc reserves loopback TCP ports for model instances.
// File locks coordinate GridCore processes before their servers bind TCP.
package portalloc

import (
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
)

// Allocator shares reservations across runtime ranges. Its methods must be
// called by one goroutine; individual reservations may be released elsewhere.
type Allocator struct {
	next  map[[2]int]int
	inUse map[int]*Reservation
	dir   string
}

// New creates an allocator. The lock directory is independent of state-dir
// and TMPDIR so a daemon and bench running as the same user coordinate.
func New() *Allocator {
	return &Allocator{
		next:  map[[2]int]int{},
		inUse: map[int]*Reservation{},
		dir:   filepath.Join("/tmp", "gridcore-ports-"+strconv.Itoa(os.Getuid())),
	}
}

// Reservation keeps a port reserved even before the child binds it.
type Reservation struct {
	Port int
	lock *os.File
	once sync.Once
}

// Release drops the process lock after the instance has stopped (or a failed
// load has finished cleanup). It is safe to call concurrently or repeatedly.
// Allocator.Release also removes the allocator's bookkeeping.
func (r *Reservation) Release() {
	if r != nil {
		r.once.Do(func() { _ = r.lock.Close() })
	}
}

// Reserve selects a port within rng, skipping local reservations, other
// GridCore processes' locks, and TCP listeners belonging to any process.
func (a *Allocator) Reserve(rng [2]int) (*Reservation, error) {
	lo, hi := rng[0], rng[1]
	if lo < 1 || hi > 65535 || lo > hi {
		return nil, fmt.Errorf("invalid port range %d-%d", lo, hi)
	}
	if err := os.MkdirAll(a.dir, 0o700); err != nil {
		return nil, fmt.Errorf("port reservations: %w", err)
	}
	next, ok := a.next[rng]
	if !ok {
		next = lo
	}
	for i := 0; i <= hi-lo; i++ {
		port := next
		next++
		if next > hi {
			next = lo
		}
		a.next[rng] = next
		if a.inUse[port] != nil {
			continue
		}
		f, err := os.OpenFile(filepath.Join(a.dir, strconv.Itoa(port)+".lock"), os.O_CREATE|os.O_RDWR, 0o600)
		if err != nil {
			return nil, fmt.Errorf("reserve port %d: %w", port, err)
		}
		if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
			_ = f.Close()
			if errors.Is(err, syscall.EWOULDBLOCK) || errors.Is(err, syscall.EAGAIN) {
				continue
			}
			return nil, fmt.Errorf("reserve port %d: %w", port, err)
		}
		ln, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
		if err != nil {
			_ = f.Close()
			if errors.Is(err, syscall.EADDRINUSE) {
				continue
			}
			return nil, fmt.Errorf("probe port %d: %w", port, err)
		}
		_ = ln.Close()
		r := &Reservation{Port: port, lock: f}
		a.inUse[port] = r
		return r, nil
	}
	return nil, fmt.Errorf("no free port in %d-%d", lo, hi)
}

// Get returns the reservation for port, or nil. A worker may retain it to
// release its lock after cleanup even if the scheduler has already exited.
func (a *Allocator) Get(port int) *Reservation { return a.inUse[port] }

// Release frees a reservation after its instance has stopped.
func (a *Allocator) Release(port int) {
	a.inUse[port].Release()
	delete(a.inUse, port)
	// Never unlink lock files: another process may already have opened the
	// same inode, and replacing it would allow two independent locks.
}

// Len returns the number of reservations owned by this allocator.
func (a *Allocator) Len() int { return len(a.inUse) }
