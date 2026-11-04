package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/w512/gridcore/internal/model"
)

func commandOutput(t *testing.T, run func() error) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "stdout")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	previous := os.Stdout
	os.Stdout = f
	defer func() { os.Stdout = previous; _ = f.Close() }()
	if err := run(); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestBenchProfilesTrackRuntimeDefaultArgs(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	stateDir := filepath.Join(dir, "state")
	writeConfig := func(layers int) {
		t.Helper()
		src := fmt.Sprintf(`
gpu: {device: simulation}
runtimes:
  sim: {type: simulation, simulated_request_delay: 1ms, default_args: ["-ngl", "%d"]}
models:
  chat: {runtime: sim, capabilities: [chat], simulated_vram_mb: 1000}
`, layers)
		if err := os.WriteFile(cfgPath, []byte(src), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	args := []string{"--config", cfgPath, "--state-dir", stateDir}
	bench := func() {
		t.Helper()
		commandOutput(t, func() error { return cmdBench(append([]string{"--all"}, args...)) })
	}
	profiles := func(sub string, extra ...string) string {
		t.Helper()
		return commandOutput(t, func() error {
			return cmdProfiles(append(append([]string{sub}, args...), extra...))
		})
	}
	stored := func() []model.Profile {
		t.Helper()
		store, err := model.OpenStore(filepath.Join(stateDir, "profiles.json"))
		if err != nil {
			t.Fatal(err)
		}
		return store.All()
	}
	writeConfig(0)
	bench()
	if out := profiles("list"); !strings.Contains(out, "current") || strings.Contains(out, "stale:") {
		t.Fatalf("bench profile not current with unchanged config:\n%s", out)
	}
	writeConfig(999)
	if out := profiles("prune", "--dry-run"); !strings.Contains(out, "would remove 1 of 1 profiles") {
		t.Errorf("old defaults did not make bench profile stale:\n%s", out)
	}
	bench()
	if all := stored(); len(all) != 2 {
		t.Fatalf("want separate profiles before and after default_args change, got %+v", all)
	}
	if out := profiles("prune"); !strings.Contains(out, "removed 1 of 2 profiles") {
		t.Errorf("prune did not remove only the outdated profile:\n%s", out)
	}
	if all := stored(); len(all) != 1 {
		t.Errorf("want one profile after prune, got %+v", all)
	}
	if out := profiles("list"); !strings.Contains(out, "current") || strings.Contains(out, "stale:") {
		t.Errorf("updated bench profile not current:\n%s", out)
	}
}
