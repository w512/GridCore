package scheduler

import (
	"fmt"
	"net"
)

// portAllocator hands out local ports from a configured range, skipping
// ports that are in use by us or by anyone else on the machine.
type portAllocator struct {
	lo, hi int
	next   int
	inUse  map[int]bool
	probe  func(port int) bool // true if free; injectable for tests
}

func newPortAllocator(lo, hi int) *portAllocator {
	return &portAllocator{lo: lo, hi: hi, next: lo, inUse: map[int]bool{}, probe: probeFree}
}

func (p *portAllocator) alloc() (int, error) {
	n := p.hi - p.lo + 1
	for i := 0; i < n; i++ {
		port := p.next
		p.next++
		if p.next > p.hi {
			p.next = p.lo
		}
		if p.inUse[port] {
			continue
		}
		if !p.probe(port) {
			continue
		}
		p.inUse[port] = true
		return port, nil
	}
	return 0, fmt.Errorf("no free port in %d-%d", p.lo, p.hi)
}

func (p *portAllocator) release(port int) { delete(p.inUse, port) }

func probeFree(port int) bool {
	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return false
	}
	ln.Close()
	return true
}
