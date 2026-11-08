package portalloc

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

func testAllocator(t *testing.T) *Allocator {
	t.Helper()
	a := New()
	a.dir = t.TempDir()
	t.Cleanup(func() {
		for port := range a.inUse {
			a.Release(port)
		}
	})
	return a
}

func TestOverlappingRangesAndRelease(t *testing.T) {
	a := testAllocator(t)
	first, err := a.Reserve([2]int{41000, 41999})
	if err != nil {
		t.Fatal(err)
	}
	overlap := [2]int{first.Port, 41999}
	additional, err := a.Reserve(overlap)
	if err != nil {
		t.Fatal(err)
	}
	if additional.Port <= first.Port || additional.Port > overlap[1] {
		t.Fatalf("overlapping range returned port %d", additional.Port)
	}
	rng := [2]int{first.Port, first.Port}
	if _, err := a.Reserve(rng); err == nil {
		t.Fatal("a different range reused a reserved port")
	}
	a.Release(first.Port)
	a.Release(first.Port) // repeated cleanup must not affect a later allocation
	second, err := a.Reserve(rng)
	if err != nil {
		t.Fatalf("port was not reusable after release: %v", err)
	}
	first.Release() // a late cleanup must not unlock the replacement
	b := testAllocator(t)
	b.dir = a.dir
	if _, err := b.Reserve(rng); err == nil {
		t.Fatal("late cleanup unlocked the replacement reservation")
	}
	a.Release(second.Port)
	if _, err := b.Reserve(rng); err != nil {
		t.Fatalf("replacement reservation was not released: %v", err)
	}
}

func TestTCPListenerIsSkippedWithoutLeakingLock(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	port := ln.Addr().(*net.TCPAddr).Port
	rng := [2]int{port, port}
	a := testAllocator(t)
	if _, err := a.Reserve(rng); err == nil {
		t.Fatal("reserved a port with an active TCP listener")
	}
	_ = ln.Close()
	b := testAllocator(t)
	b.dir = a.dir
	if _, err := b.Reserve(rng); err != nil {
		t.Fatalf("TCP probe failure leaked its file lock: %v", err)
	}
}

// A separate GridCore process reserves a port without opening a TCP listener.
// Killing it must release the lock even though its lock file stays on disk.
func TestReservationAcrossProcesses(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	a := testAllocator(t)
	rng := [2]int{port, port}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestReservationProcessHelper$")
	cmd.Env = append(os.Environ(), "GRIDCORE_TEST_PORT_LOCK_DIR="+a.dir, "GRIDCORE_TEST_PORT="+strconv.Itoa(port))
	cmd.Stderr = os.Stderr
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	input, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cmd.Process.Kill(); _ = cmd.Wait() })
	line, err := bufio.NewReader(out).ReadString('\n')
	if err != nil || strings.TrimSpace(line) != "reserved" {
		t.Fatalf("helper did not reserve port: %q, %v", line, err)
	}
	if _, err := a.Reserve(rng); err == nil {
		t.Fatal("reserved a port held by another process before TCP bind")
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = cmd.Wait()
	if _, err := a.Reserve(rng); err != nil {
		t.Fatalf("port stayed locked after reserving process died: %v", err)
	}
}

func TestReservationProcessHelper(t *testing.T) {
	dir := os.Getenv("GRIDCORE_TEST_PORT_LOCK_DIR")
	if dir == "" {
		return
	}
	port, err := strconv.Atoi(os.Getenv("GRIDCORE_TEST_PORT"))
	if err != nil {
		t.Fatal(err)
	}
	a := New()
	a.dir = dir
	if _, err := a.Reserve([2]int{port, port}); err != nil {
		t.Fatal(err)
	}
	defer a.Release(port)
	fmt.Fprintln(os.Stdout, "reserved")
	_, _ = io.Copy(io.Discard, os.Stdin)
}
