package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// Profile is what GridCore has measured about a model variant on this
// machine. Values are learned, not configured.
type Profile struct {
	Key     string `json:"key"`
	ModelID string `json:"model_id"`
	// Runtime and GPU are the runtime build and device name the key was
	// derived from. Informational (the key already encodes them); they let
	// `gridcore profiles` say why an entry no longer applies.
	Runtime string `json:"runtime,omitempty"`
	GPU     string `json:"gpu,omitempty"`

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
//
// The daemon and one-off commands (`gridcore bench`, `gridcore profiles
// prune`) may use the same file at once. Before every write the store
// therefore merges what another process changed since this store last read
// or wrote the file: entries added there are adopted, entries removed there
// are dropped, and where both sides changed an entry the newer UpdatedAt
// wins. An update that changes nothing is not written.
type Store struct {
	path string

	mu       sync.Mutex
	profiles map[string]*Profile
	disk     map[string]bool // keys in the file as of the last sync
	stamp    fileStamp       // the file as of the last sync
}

// fileStamp identifies a version of the profiles file.
type fileStamp struct {
	exists bool
	mod    int64
	size   int64
}

func statFile(path string) fileStamp {
	fi, err := os.Stat(path)
	if err != nil {
		return fileStamp{}
	}
	return fileStamp{exists: true, mod: fi.ModTime().UnixNano(), size: fi.Size()}
}

// OpenStore loads profiles from path (missing file = empty store).
func OpenStore(path string) (*Store, error) {
	s := &Store{path: path, profiles: map[string]*Profile{}, disk: map[string]bool{}}
	if path == "" {
		return s, nil
	}
	if err := s.refreshLocked(); err != nil {
		return nil, err
	}
	return s, nil
}

// refreshLocked merges changes made to the file by other processes since
// the last sync. It is a stat call when nothing changed.
func (s *Store) refreshLocked() error {
	if s.path == "" {
		return nil
	}
	st := statFile(s.path)
	if st == s.stamp {
		return nil
	}
	onDisk := map[string]*Profile{}
	raw, err := os.ReadFile(s.path)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return err
	default:
		var list []*Profile
		if err := json.Unmarshal(raw, &list); err != nil {
			return fmt.Errorf("profiles %s: %w", s.path, err)
		}
		for _, p := range list {
			onDisk[p.Key] = p
		}
	}
	for k := range s.profiles {
		if _, ok := onDisk[k]; !ok && s.disk[k] {
			delete(s.profiles, k) // removed by another process
		}
	}
	for k, dp := range onDisk {
		if mp, ok := s.profiles[k]; !ok || dp.UpdatedAt.After(mp.UpdatedAt) {
			s.profiles[k] = dp
		}
	}
	s.disk = make(map[string]bool, len(onDisk))
	for k := range onDisk {
		s.disk[k] = true
	}
	s.stamp = st
	return nil
}

// Get returns a copy of the profile for key, if any. It does not look at
// the file: changes made by other processes arrive with the next Update.
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
// persists the store if anything changed.
func (s *Store) Update(key, modelID string, fn func(*Profile)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.refreshLocked(); err != nil {
		return err
	}
	p, ok := s.profiles[key]
	if !ok {
		p = &Profile{Key: key, ModelID: modelID}
	}
	before := *p
	fn(p)
	if ok && *p == before {
		return nil
	}
	s.profiles[key] = p
	return s.saveLocked()
}

// Delete removes the given keys and persists the store.
func (s *Store) Delete(keys ...string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.refreshLocked(); err != nil {
		return err
	}
	for _, k := range keys {
		delete(s.profiles, k)
	}
	return s.saveLocked()
}

// All returns copies of all profiles, ordered by model id and then key.
func (s *Store) All() []Profile {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Profile, 0, len(s.profiles))
	for _, p := range s.profiles {
		out = append(out, *p)
	}
	sortProfiles(out)
	return out
}

func sortProfiles(l []Profile) {
	sort.Slice(l, func(i, j int) bool {
		if l[i].ModelID != l[j].ModelID {
			return l[i].ModelID < l[j].ModelID
		}
		return l[i].Key < l[j].Key
	})
}

func (s *Store) saveLocked() error {
	if s.path == "" {
		return nil // in-memory store
	}
	list := make([]Profile, 0, len(s.profiles))
	for _, p := range s.profiles {
		list = append(list, *p)
	}
	sortProfiles(list)
	raw, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	// A unique temp name: another process may be saving at the same time.
	f, err := os.CreateTemp(dir, filepath.Base(s.path)+".*.tmp")
	if err != nil {
		return err
	}
	tmp := f.Name()
	_, werr := f.Write(raw)
	cerr := f.Close()
	if err := errors.Join(werr, cerr, os.Chmod(tmp, 0o644)); err != nil {
		os.Remove(tmp)
		return err
	}
	// Rename keeps mtime and size, so this is the stamp of the file we are
	// about to publish; a later write by someone else will differ from it.
	st := statFile(tmp)
	if err := os.Rename(tmp, s.path); err != nil {
		os.Remove(tmp)
		return err
	}
	s.stamp = st
	s.disk = make(map[string]bool, len(list))
	for _, p := range list {
		s.disk[p.Key] = true
	}
	return nil
}
