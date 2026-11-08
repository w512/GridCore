package main

import (
	"context"
	"errors"
	"net"
	"strconv"
	"testing"

	"github.com/w512/gridcore/internal/config"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/portalloc"
	"github.com/w512/gridcore/internal/runtime"
)

type portCheckingRuntime struct {
	runtime.Runtime
	load func(int) error
}

func (r portCheckingRuntime) Load(_ context.Context, _ *model.Spec, port int) (runtime.Instance, error) {
	return nil, r.load(port)
}

func TestBenchReservesPortUntilLoadCleanup(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	rng := [2]int{port, port}
	cfg := config.Default()
	cfg.Runtimes["test"] = config.Runtime{PortRange: rng}
	sp := &model.Spec{ID: "test", Runtime: "test"}
	other := portalloc.New()
	loadErr := errors.New("intentional load failure")
	called := false
	rt := portCheckingRuntime{load: func(got int) error {
		called = true
		if got != port {
			t.Fatalf("load port = %d, want %d", got, port)
		}
		// No TCP socket is bound while a runtime initialises. The file lock
		// must keep another daemon/bench from selecting this port anyway.
		probe, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
		if err != nil {
			t.Fatal(err)
		}
		_ = probe.Close()
		if reservation, err := other.Reserve(rng); err == nil {
			other.Release(reservation.Port)
			t.Error("bench port was not reserved during load")
		}
		return loadErr
	}}
	result := benchOne(context.Background(), cfg, rt, nil, sp, 1, 1)
	if !called || !errors.Is(result.Err, loadErr) {
		t.Fatalf("bench did not reach runtime load: %v", result.Err)
	}
	reservation, err := other.Reserve(rng)
	if err != nil {
		t.Fatalf("failed bench load leaked its port reservation: %v", err)
	}
	other.Release(reservation.Port)
}
