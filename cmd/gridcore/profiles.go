package main

import (
	"context"
	"flag"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/scheduler"
)

// profileStatus says whether a stored profile is the one the scheduler
// would use for a configured model right now.
type profileStatus struct {
	current bool
	reason  string // why not, when !current
}

// cmdProfiles lists measured profiles (`gridcore profiles`) or removes the
// ones no configured model can use any more (`gridcore profiles prune`).
// Safe while the daemon runs: the store merges concurrent changes.
func cmdProfiles(args []string) error {
	sub := "list"
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		sub, args = args[0], args[1:]
	}
	if sub != "list" && sub != "prune" {
		return fmt.Errorf("profiles: unknown subcommand %q (want list or prune)", sub)
	}
	fs := flag.NewFlagSet("profiles "+sub, flag.ExitOnError)
	cfgPath := configFlag(fs)
	stateDir := fs.String("state-dir", "", "state directory (default $XDG_STATE_HOME/gridcore)")
	dryRun := fs.Bool("dry-run", false, "prune: only show what would be removed")
	_ = fs.Parse(args)

	cfg, err := loadConfig(*cfgPath, *stateDir)
	if err != nil {
		return err
	}
	path := filepath.Join(cfg.StateDir, "profiles.json")
	store, err := model.OpenStore(path)
	if err != nil {
		return err
	}

	// The key includes the runtime build and the GPU name; ask both the
	// same way the daemon does.
	mon, runtimes, err := buildBackends(cfg, "")
	if err != nil {
		return err
	}
	gpuName := ""
	if snap, err := mon.Snapshot(context.Background()); err == nil {
		gpuName = snap.Name
	} else if sub == "prune" {
		return fmt.Errorf("cannot tell which GPU profiles belong to: %w", err)
	}
	rtIDs := map[string]string{}
	for name, rc := range cfg.Runtimes {
		rtIDs[name] = scheduler.RuntimeID(rc, runtimes[name])
	}
	specs := model.FromConfig(cfg)
	statusOf := func(p model.Profile) profileStatus {
		sp, ok := specs[p.ModelID]
		if !ok {
			return profileStatus{reason: "model not in config"}
		}
		rtID := rtIDs[sp.Runtime]
		switch {
		case p.Key == sp.ProfileKey(rtID, gpuName):
			return profileStatus{current: true}
		case gpuName == "":
			return profileStatus{reason: "unknown (no GPU reading)"}
		case p.Runtime != "" && p.Runtime != rtID:
			return profileStatus{reason: "runtime build " + p.Runtime}
		case p.GPU != "" && p.GPU != gpuName:
			return profileStatus{reason: "other GPU " + p.GPU}
		default:
			return profileStatus{reason: "model settings or runtime changed"}
		}
	}

	all := store.All()
	if sub == "list" {
		printProfiles(all, statusOf)
		return nil
	}

	var stale []string
	for _, p := range all {
		if st := statusOf(p); !st.current {
			stale = append(stale, p.Key)
			fmt.Printf("%-18s %s  %s\n", p.ModelID, p.Key, st.reason)
		}
	}
	switch {
	case len(stale) == 0:
		fmt.Println("nothing to prune")
	case *dryRun:
		fmt.Printf("would remove %d of %d profiles from %s\n", len(stale), len(all), path)
	default:
		if err := store.Delete(stale...); err != nil {
			return err
		}
		fmt.Printf("removed %d of %d profiles from %s\n", len(stale), len(all), path)
	}
	return nil
}

func printProfiles(all []model.Profile, statusOf func(model.Profile) profileStatus) {
	if len(all) == 0 {
		fmt.Println("no profiles yet: they are recorded when a model loads (or by `gridcore bench`)")
		return
	}
	fmt.Printf("%-18s %8s %8s %8s  %-16s  %s\n", "MODEL", "VRAM MB", "LOAD ms", "GEN t/s", "UPDATED", "STATUS")
	current := 0
	for _, p := range all {
		st := statusOf(p)
		status := "current"
		if st.current {
			current++
		} else {
			status = "stale: " + st.reason
		}
		updated := "-"
		if !p.UpdatedAt.IsZero() {
			updated = p.UpdatedAt.Local().Format("2006-01-02 15:04")
		}
		fmt.Printf("%-18s %8d %8.0f %8.1f  %-16s  %s\n", p.ModelID, p.VRAMMB, p.LoadMS, p.GenTPS, updated, status)
	}
	if n := len(all) - current; n > 0 {
		fmt.Printf("\n%d of %d profiles are stale; `gridcore profiles prune` removes them\n", n, len(all))
	}
}
