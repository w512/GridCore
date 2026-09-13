package fake

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"sync"
	"testing"
	"time"

	gpufake "github.com/w512/gridcore/internal/gpu/fake"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/runtime"
)

func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

func TestLoadServeStop(t *testing.T) {
	gpu := gpufake.New("fake", 16000)
	rt := New("sim", gpu)
	rt.RequestDelay = 5 * time.Millisecond
	spec := &model.Spec{ID: "m", RuntimeType: "fake", FakeVRAMMB: 9000, FakeLoad: 10 * time.Millisecond}

	ctx := context.Background()
	start := time.Now()
	inst, err := rt.Load(ctx, spec, freePort(t))
	if err != nil {
		t.Fatal(err)
	}
	if time.Since(start) < spec.FakeLoad {
		t.Error("load should take at least fake_load_time")
	}
	if err := inst.Health(ctx); err != nil {
		t.Fatalf("health: %v", err)
	}

	snap, _ := gpu.Snapshot(ctx)
	if snap.UsedByPID(inst.PID()) != 9000 || snap.UsedMB != 9000 {
		t.Errorf("gpu should see 9000MB for pid %d: %+v", inst.PID(), snap)
	}

	// Chat, non-streaming.
	body := bytes.NewBufferString(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	resp, err := http.Post("http://"+inst.Addr()+"/v1/chat/completions", "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Choices []struct {
			Message struct{ Content string } `json:"message"`
		} `json:"choices"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()
	if len(out.Choices) != 1 || out.Choices[0].Message.Content != "ok from m" {
		t.Errorf("chat response = %+v", out)
	}

	// Embeddings with 3 inputs.
	body = bytes.NewBufferString(`{"model":"m","input":["a","b","c"]}`)
	resp, err = http.Post("http://"+inst.Addr()+"/v1/embeddings", "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	var emb struct {
		Data []struct{ Index int } `json:"data"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&emb)
	resp.Body.Close()
	if len(emb.Data) != 3 || emb.Data[2].Index != 2 {
		t.Errorf("embeddings = %+v", emb)
	}

	if err := inst.Stop(ctx); err != nil {
		t.Fatalf("stop: %v", err)
	}
	select {
	case <-inst.Done():
	case <-time.After(time.Second):
		t.Fatal("Done not closed after Stop")
	}
	snap, _ = gpu.Snapshot(ctx)
	if snap.UsedMB != 0 {
		t.Errorf("gpu should be empty after stop: %+v", snap)
	}
	if inst.Health(ctx) == nil {
		t.Error("health should fail after stop")
	}
}

func TestConcurrencyTracking(t *testing.T) {
	rt := New("sim", nil)
	rt.RequestDelay = 30 * time.Millisecond
	spec := &model.Spec{ID: "m", RuntimeType: "fake"}
	inst, err := rt.Load(context.Background(), spec, freePort(t))
	if err != nil {
		t.Fatal(err)
	}
	defer inst.Stop(context.Background())

	var wg sync.WaitGroup
	for n := 0; n < 4; n++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := http.Post("http://"+inst.Addr()+"/v1/chat/completions", "application/json", bytes.NewBufferString(`{}`))
			if err == nil {
				resp.Body.Close()
			}
		}()
	}
	wg.Wait()
	fi := inst.(*Instance)
	if fi.Requests() != 4 || fi.PeakConcurrency() < 2 {
		t.Errorf("requests=%d peak=%d", fi.Requests(), fi.PeakConcurrency())
	}
}

func TestLoadCancelled(t *testing.T) {
	rt := New("sim", nil)
	spec := &model.Spec{ID: "m", RuntimeType: "fake", FakeLoad: time.Second}
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(10 * time.Millisecond); cancel() }()
	_, err := rt.Load(ctx, spec, freePort(t))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

func TestLoadTimeout(t *testing.T) {
	rt := New("sim", nil)
	rt.LoadTimeout = 10 * time.Millisecond
	spec := &model.Spec{ID: "m", RuntimeType: "fake", FakeLoad: time.Second}
	_, err := rt.Load(context.Background(), spec, freePort(t))
	if !errors.Is(err, runtime.ErrLoadTimeout) {
		t.Fatalf("want ErrLoadTimeout, got %v", err)
	}
}

func TestKillClosesDone(t *testing.T) {
	gpu := gpufake.New("fake", 16000)
	rt := New("sim", gpu)
	spec := &model.Spec{ID: "m", RuntimeType: "fake", FakeVRAMMB: 100}
	inst, err := rt.Load(context.Background(), spec, freePort(t))
	if err != nil {
		t.Fatal(err)
	}
	inst.(*Instance).Kill()
	select {
	case <-inst.Done():
	case <-time.After(time.Second):
		t.Fatal("Done not closed after Kill")
	}
	if inst.Err() == nil {
		t.Error("Err should be non-nil after Kill")
	}
	snap, _ := gpu.Snapshot(context.Background())
	if snap.UsedMB != 0 {
		t.Error("killed process should release VRAM")
	}
}
