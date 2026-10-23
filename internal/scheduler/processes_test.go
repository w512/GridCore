package scheduler

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestGPUProcessesListsEveryProcess(t *testing.T) {
	h := newHarness(t, 16000, chat+pinned)
	h.eventually(isReady("embed"), "pinned loaded")
	if err := h.s.LoadModel("chat"); err != nil {
		t.Fatal(err)
	}
	h.eventually(isReady("chat"), "chat loaded")
	chatPID, embedPID := h.resident("chat").PID, h.resident("embed").PID

	// PIDs below the simulation's (100000+) are someone else's processes.
	h.names.set(1337, "Xorg")
	h.names.set(4242, "game")
	h.gpu.Attach(1337, 310)
	h.gpu.Attach(4242, 4000)
	h.gpu.Attach(1200, 310) // name unknown: gone, or another user's
	h.eventually(func(st State) bool { return len(st.GPU.Processes) == 5 }, "external processes listed")

	st := h.s.State()
	want := []GPUProcess{
		{PID: chatPID, Name: "llama-server", UsedMB: 9000, Model: "chat"},
		{PID: 4242, Name: "game", UsedMB: 4000},
		{PID: embedPID, Name: "llama-server", UsedMB: 600, Model: "embed"},
		{PID: 1200, Name: "", UsedMB: 310}, // ties by pid
		{PID: 1337, Name: "Xorg", UsedMB: 310},
	}
	for i, w := range want {
		if st.GPU.Processes[i] != w {
			t.Errorf("processes[%d] = %+v, want %+v", i, st.GPU.Processes[i], w)
		}
	}
	if st.GPU.ExternalMB != 4620 {
		t.Errorf("external = %d, want 4620", st.GPU.ExternalMB)
	}

	// The GPU is polled every 10ms here; names are looked up once per PID.
	time.Sleep(100 * time.Millisecond)
	for _, pid := range []int{chatPID, embedPID, 1337, 4242, 1200} {
		if n := h.names.lookups(pid); n != 1 {
			t.Errorf("pid %d looked up %d times, want 1", pid, n)
		}
	}

	// A PID that leaves and comes back may be a different process.
	h.gpu.Detach(4242)
	h.eventually(func(st State) bool { return len(st.GPU.Processes) == 4 }, "game gone")
	h.names.set(4242, "llama-server")
	h.gpu.Attach(4242, 7940)
	h.eventually(func(st State) bool {
		return len(st.GPU.Processes) == 5 && st.GPU.Processes[1] == GPUProcess{PID: 4242, Name: "llama-server", UsedMB: 7940}
	}, "a reused PID gets its new name")
}

func TestGPUProcessesJSON(t *testing.T) {
	h := newHarness(t, 16000, chat+pinned)
	h.eventually(isReady("embed"), "pinned loaded")
	h.names.set(1337, "Xorg")
	h.gpu.Attach(1337, 310)
	h.eventually(func(st State) bool { return len(st.GPU.Processes) == 2 }, "Xorg listed")

	b, err := json.Marshal(h.s.State())
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		GPU struct {
			Processes []map[string]any `json:"processes"`
		} `json:"gpu"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	ps := doc.GPU.Processes
	if len(ps) != 2 || ps[0]["model"] != "embed" || ps[0]["used_mb"] != 600.0 {
		t.Fatalf("processes = %v", ps)
	}
	if _, ok := ps[1]["model"]; ok || ps[1]["name"] != "Xorg" || ps[1]["pid"] != 1337.0 || ps[1]["used_mb"] != 310.0 {
		t.Errorf("an external process has no model: %v", ps[1])
	}

	// No processes at all, and states built without a snapshot: [] not null.
	for _, st := range []State{{}, {GPU: GPUState{Processes: []GPUProcess{}}}} {
		b, err := json.Marshal(st)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(b), `"processes":[]`) {
			t.Errorf("want an empty array: %s", b)
		}
	}
}
