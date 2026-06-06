package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Profile is what GridCore has measured about a model variant on this
// machine. Values are learned, not configured.
type Profile struct {
	Key     string `json:"key"`
	ModelID string `json:"model_id"`

	VRAMMB    int     `json:"vram_mb"`    // max per-process VRAM observed while resident
	LoadMS    float64 `json:"load_ms"`    // EMA of spawn -> healthy
	GenTPS    float64 `json:"gen_tps"`    // EMA generation tokens/s
	PromptTPS float64 `json:"prompt_tps"` // EMA prompt tokens/s
	Samples   int     `json:"samples"`

	UpdatedAt time.Time `json:"updated_at"`
}

// emaAlpha weights new observations; ~last 10 samples matter.
const emaAlpha = 0.2

func ema(prev, obs float64, samples int) float64 {
	if samples == 0 || prev == 0 {
		return obs
	}
	return prev*(1-emaAlpha) + obs*emaAlpha
}

// ObserveLoad records a successful load duration.
func (p *Profile) ObserveLoad(d time.Duration) {
	p.LoadMS = ema(p.LoadMS, float64(d.Milliseconds()), p.Samples)
	p.Samples++
	p.UpdatedAt = time.Now()
}

// ObserveVRAM records a per-process VRAM reading; keeps the max.
func (p *Profile) ObserveVRAM(mb int) {
	if mb > p.VRAMMB {
		p.VRAMMB = mb
		p.UpdatedAt = time.Now()
	}
}

// ObserveThroughput records tokens/s from a finished request. Zero values
// are ignored so partial usage info does not drag the average down.
func (p *Profile) ObserveThroughput(promptTPS, genTPS float64) {
	if promptTPS > 0 {
		p.PromptTPS = ema(p.PromptTPS, promptTPS, p.Samples)
	}
	if genTPS > 0 {
		p.GenTPS = ema(p.GenTPS, genTPS, p.Samples)
	}
	p.UpdatedAt = time.Now()
}

// Store persists Profiles as a single JSON file with atomic writes.
type Store struct {
	path string

	mu       sync.Mutex
	profiles map[string]*Profile
}

// OpenStore loads profiles from path (missing file = empty store).
func OpenStore(path string) (*Store, error) {
	s := &Store{path: path, profiles: map[string]*Profile{}}
	raw, err := os.ReadFile(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		return s, nil
	case err != nil:
		return nil, err
	}
	var list []*Profile
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("profiles %s: %w", path, err)
	}
	for _, p := range list {
		s.profiles[p.Key] = p
	}
	return s, nil
}

// Get returns a copy of the profile for key, if any.
func (s *Store) Get(key string) (Profile, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p, ok := s.profiles[key]
	if !ok {
		return Profile{}, false
	}
	return *p, true
}

// Update applies fn to the profile for key (creating it if needed) and
// persists the store.
func (s *Store) Update(key, modelID string, fn func(*Profile)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	p, ok := s.profiles[key]
	if !ok {
		p = &Profile{Key: key, ModelID: modelID}
		s.profiles[key] = p
	}
	fn(p)
	return s.saveLocked()
}

// All returns copies of all profiles.
func (s *Store) All() []Profile {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Profile, 0, len(s.profiles))
	for _, p := range s.profiles {
		out = append(out, *p)
	}
	return out
}

func (s *Store) saveLocked() error {
	if s.path == "" {
		return nil // in-memory store
	}
	list := make([]*Profile, 0, len(s.profiles))
	for _, p := range s.profiles {
		list = append(list, p)
	}
	raw, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}
