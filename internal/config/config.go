// Package config loads, defaults and validates the GridCore YAML config.
//
// The config is pure data: it does not touch the filesystem beyond reading
// the file itself. Path existence is checked separately via CheckFiles so
// that tests and the fake runtime can run without any model files.
package config

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/gridcore/gridcore/internal/job"
)

// Runtime types known to the process supervisor.
const (
	RuntimeLlamaCpp = "llamacpp"
	RuntimeFake     = "fake"
)

// Capabilities a model may declare. They gate which endpoints may target it.
const (
	CapChat       = "chat"
	CapCompletion = "completion"
	CapEmbedding  = "embedding"
	CapVision     = "vision"
)

var knownCapabilities = map[string]bool{
	CapChat: true, CapCompletion: true, CapEmbedding: true, CapVision: true,
}

// Config is the root configuration.
type Config struct {
	Server   Server             `yaml:"server"`
	GPU      GPU                `yaml:"gpu"`
	Runtimes map[string]Runtime `yaml:"runtimes"`
	Models   map[string]Model   `yaml:"models"`
	Policy   Policy             `yaml:"policy"`

	// StateDir is where profiles.json and child logs live. Not set from YAML;
	// resolved from $XDG_STATE_HOME or --state-dir.
	StateDir string `yaml:"-"`
}

type Server struct {
	Listen string `yaml:"listen"`
}

type GPU struct {
	// Device selects the monitor: "auto", "nvidia:<idx>" or "fake".
	Device string `yaml:"device"`
	// VRAMLimitMB caps the budget below the physical total. nil = no cap.
	VRAMLimitMB *int `yaml:"vram_limit_mb"`
	// HeadroomMB is subtracted from the budget to absorb fragmentation and
	// compute buffers not attributed to a process.
	HeadroomMB   int           `yaml:"headroom_mb"`
	PollInterval time.Duration `yaml:"poll_interval"`
	// FakeTotalMB is the capacity of the fake device (device: fake).
	FakeTotalMB int `yaml:"fake_total_mb"`
}

type Runtime struct {
	// Type selects the adapter (llamacpp | fake). Defaults to the map key.
	Type        string        `yaml:"type"`
	Binary      string        `yaml:"binary"`
	PortRange   [2]int        `yaml:"port_range"`
	LoadTimeout time.Duration `yaml:"load_timeout"`
	DefaultArgs []string      `yaml:"default_args"`
	// FakeRequestDelay is how long a fake instance takes per request.
	FakeRequestDelay time.Duration `yaml:"fake_request_delay"`
}

type Model struct {
	Runtime      string   `yaml:"runtime"`
	Path         string   `yaml:"path"`
	MMProj       string   `yaml:"mmproj"`
	Capabilities []string `yaml:"capabilities"`
	// Ctx is the total context pool shared by all slots (llama-server
	// --ctx-size). Parallel is the slot count (--parallel) and therefore the
	// model's request concurrency.
	Ctx      int      `yaml:"ctx"`
	Parallel int      `yaml:"parallel"`
	Args     []string `yaml:"args"`
	Aliases  []string `yaml:"aliases"`
	// Pinned models are preloaded at start and never evicted.
	Pinned bool `yaml:"pinned"`
	// Preload loads the model at start but leaves it evictable.
	Preload bool `yaml:"preload"`

	// Fake-runtime knobs (ignored by real runtimes). Let tests and demos
	// describe a model's footprint without a file on disk.
	FakeVRAMMB int           `yaml:"fake_vram_mb"`
	FakeLoad   time.Duration `yaml:"fake_load_time"`
}

type Policy struct {
	DefaultClass                    job.Class                 `yaml:"default_class"`
	Classes                         map[job.Class]ClassPolicy `yaml:"classes"`
	InteractiveIdleBeforeBackground time.Duration             `yaml:"interactive_idle_before_background"`
	// BackgroundMaxStarvation bounds how long background/batch work can be
	// held back by continuous interactive traffic. Once a background job
	// has made no progress for this long, one background step at a time may
	// run alongside interactive work. Unset = 3s; "0s" disables the
	// guarantee (strict exclusivity). A pointer so that 0 and unset differ.
	BackgroundMaxStarvation *time.Duration `yaml:"background_max_starvation"`
	EmbeddingChunkSize      int            `yaml:"embedding_chunk_size"`
}

// MaxStarvation returns the effective background_max_starvation.
func (p Policy) MaxStarvation() time.Duration {
	if p.BackgroundMaxStarvation == nil {
		return 3 * time.Second
	}
	return *p.BackgroundMaxStarvation
}

type ClassPolicy struct {
	Priority int `yaml:"priority"`
	// HotTTL only applies to interactive: how long after the last interactive
	// use a model stays "hot" (protected from eviction by lower classes).
	HotTTL time.Duration `yaml:"hot_ttl"`
}

// Default returns a config with every default applied and no models.
func Default() *Config {
	c := &Config{}
	c.applyDefaults()
	return c
}

// Load reads, defaults and validates a config file.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(raw)
}

// Parse decodes YAML bytes, applies defaults and validates.
func Parse(raw []byte) (*Config, error) {
	var c Config
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true) // typos in keys are errors, not silent no-ops
	if err := dec.Decode(&c); err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	c.applyDefaults()
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

func (c *Config) applyDefaults() {
	if c.Server.Listen == "" {
		c.Server.Listen = "127.0.0.1:8080"
	}
	if c.GPU.Device == "" {
		c.GPU.Device = "auto"
	}
	if c.GPU.HeadroomMB == 0 {
		c.GPU.HeadroomMB = 512
	}
	if c.GPU.PollInterval == 0 {
		c.GPU.PollInterval = 500 * time.Millisecond
	}
	if c.GPU.FakeTotalMB == 0 {
		c.GPU.FakeTotalMB = 16384
	}

	if c.Runtimes == nil {
		c.Runtimes = map[string]Runtime{}
	}
	for name, rt := range c.Runtimes {
		if rt.Type == "" {
			rt.Type = name
		}
		if rt.PortRange == [2]int{} {
			rt.PortRange = [2]int{41000, 41999}
		}
		if rt.LoadTimeout == 0 {
			rt.LoadTimeout = 120 * time.Second
		}
		if rt.Type == RuntimeFake && rt.FakeRequestDelay == 0 {
			rt.FakeRequestDelay = 50 * time.Millisecond
		}
		c.Runtimes[name] = rt
	}

	if c.Models == nil {
		c.Models = map[string]Model{}
	}
	for id, m := range c.Models {
		if m.Ctx == 0 {
			m.Ctx = 4096
		}
		if m.Parallel == 0 {
			m.Parallel = 1
		}
		if m.Pinned {
			m.Preload = true
		}
		m.Path = expandHome(m.Path)
		m.MMProj = expandHome(m.MMProj)
		c.Models[id] = m
	}

	if c.Policy.DefaultClass == "" {
		c.Policy.DefaultClass = job.Interactive
	}
	if c.Policy.Classes == nil {
		c.Policy.Classes = map[job.Class]ClassPolicy{}
	}
	defaults := map[job.Class]ClassPolicy{
		job.Interactive: {Priority: 100, HotTTL: 5 * time.Minute},
		job.Background:  {Priority: 30},
		job.Batch:       {Priority: 10},
	}
	for cls, d := range defaults {
		p, ok := c.Policy.Classes[cls]
		if !ok {
			c.Policy.Classes[cls] = d
			continue
		}
		if p.Priority == 0 {
			p.Priority = d.Priority
		}
		if p.HotTTL == 0 {
			p.HotTTL = d.HotTTL
		}
		c.Policy.Classes[cls] = p
	}
	if c.Policy.InteractiveIdleBeforeBackground == 0 {
		c.Policy.InteractiveIdleBeforeBackground = 2 * time.Second
	}
	if c.Policy.EmbeddingChunkSize == 0 {
		c.Policy.EmbeddingChunkSize = 32
	}
}

// Validate checks internal consistency. It never touches the filesystem.
func (c *Config) Validate() error {
	var errs []error
	add := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }

	if c.GPU.HeadroomMB < 0 {
		add("gpu.headroom_mb must be >= 0")
	}
	if c.GPU.VRAMLimitMB != nil && *c.GPU.VRAMLimitMB <= 0 {
		add("gpu.vram_limit_mb must be > 0 when set")
	}
	if !validDevice(c.GPU.Device) {
		add("gpu.device %q: want auto, nvidia:<index> or fake", c.GPU.Device)
	}

	for name, rt := range c.Runtimes {
		switch rt.Type {
		case RuntimeLlamaCpp:
			if rt.Binary == "" {
				add("runtimes.%s.binary is required", name)
			}
		case RuntimeFake:
		default:
			add("runtimes.%s.type %q: want llamacpp or fake", name, rt.Type)
		}
		lo, hi := rt.PortRange[0], rt.PortRange[1]
		if lo < 1024 || hi > 65535 || lo > hi {
			add("runtimes.%s.port_range [%d, %d] is invalid", name, lo, hi)
		}
		if rt.LoadTimeout <= 0 {
			add("runtimes.%s.load_timeout must be > 0", name)
		}
	}

	// Names (ids + aliases) must be unique across all models.
	seen := map[string]string{} // name -> owning model id
	ids := sortedKeys(c.Models)
	for _, id := range ids {
		m := c.Models[id]
		if strings.Contains(id, "@") {
			add("models.%s: id must not contain '@' (reserved for class suffix)", id)
		}
		rt, ok := c.Runtimes[m.Runtime]
		if !ok {
			add("models.%s.runtime %q is not defined in runtimes", id, m.Runtime)
		}
		if m.Path == "" && rt.Type != RuntimeFake {
			add("models.%s.path is required", id)
		}
		if ok && rt.Type == RuntimeFake && m.FakeVRAMMB <= 0 {
			add("models.%s.fake_vram_mb must be > 0 for fake runtimes", id)
		}
		if len(m.Capabilities) == 0 {
			add("models.%s.capabilities must not be empty", id)
		}
		for _, cap := range m.Capabilities {
			if !knownCapabilities[cap] {
				add("models.%s.capabilities: unknown %q", id, cap)
			}
		}
		if m.Ctx <= 0 {
			add("models.%s.ctx must be > 0", id)
		}
		if m.Parallel <= 0 {
			add("models.%s.parallel must be > 0", id)
		}
		if prev, dup := seen[id]; dup {
			add("models.%s: name already used by %s", id, prev)
		}
		seen[id] = id
		for _, a := range m.Aliases {
			if strings.Contains(a, "@") {
				add("models.%s.aliases: %q must not contain '@'", id, a)
			}
			if prev, dup := seen[a]; dup {
				add("models.%s.aliases: %q already used by %s", id, a, prev)
				continue
			}
			seen[a] = id
		}
	}

	if _, ok := c.Policy.Classes[c.Policy.DefaultClass]; !ok {
		add("policy.default_class %q is not a known class", c.Policy.DefaultClass)
	}
	for cls := range c.Policy.Classes {
		if _, err := job.ParseClass(string(cls)); err != nil {
			add("policy.classes: %v", err)
		}
	}
	if c.Policy.EmbeddingChunkSize <= 0 {
		add("policy.embedding_chunk_size must be > 0")
	}
	if c.Policy.InteractiveIdleBeforeBackground < 0 {
		add("policy.interactive_idle_before_background must be >= 0")
	}
	if c.Policy.BackgroundMaxStarvation != nil && *c.Policy.BackgroundMaxStarvation < 0 {
		add("policy.background_max_starvation must be >= 0")
	}

	return errors.Join(errs...)
}

// CheckFiles verifies that runtime binaries and model files exist. Fake
// runtimes are skipped. Call this from `serve`, not from tests.
func (c *Config) CheckFiles() error {
	var errs []error
	for name, rt := range c.Runtimes {
		if rt.Type == RuntimeFake {
			continue
		}
		if _, err := os.Stat(rt.Binary); err != nil {
			errs = append(errs, fmt.Errorf("runtimes.%s.binary: %w", name, err))
		}
	}
	for id, m := range c.Models {
		if c.Runtimes[m.Runtime].Type == RuntimeFake {
			continue
		}
		if _, err := os.Stat(m.Path); err != nil {
			errs = append(errs, fmt.Errorf("models.%s.path: %w", id, err))
		}
		if m.MMProj != "" {
			if _, err := os.Stat(m.MMProj); err != nil {
				errs = append(errs, fmt.Errorf("models.%s.mmproj: %w", id, err))
			}
		}
	}
	return errors.Join(errs...)
}

// ResolveModel maps a user-facing name (id or alias) to a model id.
func (c *Config) ResolveModel(name string) (string, bool) {
	if _, ok := c.Models[name]; ok {
		return name, true
	}
	for id, m := range c.Models {
		for _, a := range m.Aliases {
			if a == name {
				return id, true
			}
		}
	}
	return "", false
}

// HasCapability reports whether model id declares cap.
func (m Model) HasCapability(cap string) bool {
	for _, c := range m.Capabilities {
		if c == cap {
			return true
		}
	}
	return false
}

// DefaultConfigPath returns $XDG_CONFIG_HOME/gridcore/config.yaml.
func DefaultConfigPath() string {
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, _ := os.UserHomeDir()
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "gridcore", "config.yaml")
}

// DefaultStateDir returns $XDG_STATE_HOME/gridcore.
func DefaultStateDir() string {
	base := os.Getenv("XDG_STATE_HOME")
	if base == "" {
		home, _ := os.UserHomeDir()
		base = filepath.Join(home, ".local", "state")
	}
	return filepath.Join(base, "gridcore")
}

func validDevice(d string) bool {
	if d == "auto" || d == "fake" {
		return true
	}
	if idx, ok := strings.CutPrefix(d, "nvidia:"); ok {
		if idx == "" {
			return false
		}
		for _, r := range idx {
			if r < '0' || r > '9' {
				return false
			}
		}
		return true
	}
	return false
}

func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[1:])
		}
	}
	return p
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
