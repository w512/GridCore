package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCheckRejectsNegativePollInterval(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte("gpu: {device: simulation, poll_interval: -1s}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := cmdCheck([]string{"--config", path}); err == nil || !strings.Contains(err.Error(), "gpu.poll_interval must be > 0") {
		t.Fatalf("check accepted an interval that would panic the scheduler: %v", err)
	}
}
