// Command gridcore is a workload scheduler for local AI computers.
//
//	gridcore serve   [--config PATH] [--state-dir DIR]   run the daemon
//	gridcore check   [--config PATH]                     validate config and files
//	gridcore status  [--addr HOST:PORT] [--watch]        show scheduler state
//	gridcore models  [--config PATH]                     list configured models
//	gridcore bench   [--all] <model-id>...               measure VRAM, load time, throughput
//	gridcore version
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/gridcore/gridcore/internal/api"
	"github.com/gridcore/gridcore/internal/config"
	"github.com/gridcore/gridcore/internal/metrics"
	"github.com/gridcore/gridcore/internal/model"
	"github.com/gridcore/gridcore/internal/scheduler"
)

// version is set by the linker (see Makefile).
var version = "dev"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "serve":
		err = cmdServe(os.Args[2:])
	case "check":
		err = cmdCheck(os.Args[2:])
	case "status":
		err = cmdStatus(os.Args[2:])
	case "models":
		err = cmdModels(os.Args[2:])
	case "bench":
		err = cmdBench(os.Args[2:])
	case "version", "--version", "-v":
		fmt.Println("gridcore", version)
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "gridcore: unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "gridcore:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `gridcore — a workload scheduler for local AI computers

Usage:
  gridcore serve   [--config PATH] [--state-dir DIR] [--log-level LEVEL]
  gridcore check   [--config PATH]
  gridcore status  [--addr HOST:PORT] [--watch] [--json]
  gridcore models  [--config PATH] [--explain]
  gridcore bench   [--config PATH] [--all] [--prompt N] [--gen N] <model-id>...
  gridcore version

Config defaults to $XDG_CONFIG_HOME/gridcore/config.yaml (or $GRIDCORE_CONFIG).
`)
}

func configFlag(fs *flag.FlagSet) *string {
	def := os.Getenv("GRIDCORE_CONFIG")
	if def == "" {
		def = config.DefaultConfigPath()
	}
	return fs.String("config", def, "path to config.yaml")
}

func loadConfig(path, stateDir string) (*config.Config, error) {
	cfg, err := config.Load(path)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", path, err)
	}
	if stateDir == "" {
		stateDir = config.DefaultStateDir()
	}
	cfg.StateDir = stateDir
	return cfg, nil
}

func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	cfgPath := configFlag(fs)
	stateDir := fs.String("state-dir", "", "state directory (default $XDG_STATE_HOME/gridcore)")
	logLevel := fs.String("log-level", "info", "debug|info|warn|error")
	_ = fs.Parse(args)

	setupLogging(*logLevel)
	cfg, err := loadConfig(*cfgPath, *stateDir)
	if err != nil {
		return err
	}
	if err := cfg.CheckFiles(); err != nil {
		return fmt.Errorf("config files: %w", err)
	}
	if err := os.MkdirAll(filepath.Join(cfg.StateDir, "logs"), 0o755); err != nil {
		return fmt.Errorf("state dir: %w", err)
	}

	m := metrics.New()
	store, err := model.OpenStore(filepath.Join(cfg.StateDir, "profiles.json"))
	if err != nil {
		return fmt.Errorf("profiles: %w", err)
	}
	mon, runtimes, err := buildBackends(cfg, cfg.StateDir)
	if err != nil {
		return err
	}
	sched, err := scheduler.New(cfg, runtimes, mon, store, m, scheduler.Options{})
	if err != nil {
		return err
	}
	srv := api.New(cfg, m, sched, version)

	httpSrv := &http.Server{
		Addr:              cfg.Server.Listen,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: streaming responses and long generations are normal.
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	schedDone := make(chan struct{})
	go func() {
		_ = sched.Run(ctx)
		close(schedDone)
	}()

	errCh := make(chan error, 1)
	go func() {
		slog.Info("gridcore listening", "addr", cfg.Server.Listen, "models", len(cfg.Models), "gpu", cfg.GPU.Device, "state_dir", cfg.StateDir)
		if err := httpSrv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		stop()
		<-schedDone
		return err
	case <-ctx.Done():
	}
	slog.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
	defer cancel()
	// Scheduler first: it drains in-flight steps and stops instances while
	// the HTTP server keeps those responses open.
	select {
	case <-schedDone:
	case <-shutdownCtx.Done():
	}
	return httpSrv.Shutdown(shutdownCtx)
}

func cmdCheck(args []string) error {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	cfgPath := configFlag(fs)
	_ = fs.Parse(args)

	cfg, err := loadConfig(*cfgPath, "")
	if err != nil {
		return err
	}
	fmt.Printf("config: ok (%d runtimes, %d models)\n", len(cfg.Runtimes), len(cfg.Models))
	if err := cfg.CheckFiles(); err != nil {
		return fmt.Errorf("files:\n%w", err)
	}
	fmt.Println("files:  ok")
	if _, _, err := buildBackends(cfg, ""); err != nil {
		return fmt.Errorf("backends: %w", err)
	}
	fmt.Println("gpu:    ok")
	return nil
}

func cmdModels(args []string) error {
	fs := flag.NewFlagSet("models", flag.ExitOnError)
	cfgPath := configFlag(fs)
	explain := fs.Bool("explain", false, "show the VRAM estimate breakdown per model (reads GGUF headers)")
	_ = fs.Parse(args)

	cfg, err := loadConfig(*cfgPath, "")
	if err != nil {
		return err
	}
	ids := make([]string, 0, len(cfg.Models))
	for id := range cfg.Models {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	specs := model.FromConfig(cfg)
	fmt.Printf("%-16s %-12s %-22s %6s %4s %8s  %s\n", "ID", "RUNTIME", "CAPABILITIES", "CTX", "PAR", "EST MB", "FLAGS")
	for _, id := range ids {
		m := cfg.Models[id]
		flags := ""
		if m.Pinned {
			flags += "pinned "
		} else if m.Preload {
			flags += "preload "
		}
		if len(m.Aliases) > 0 {
			flags += fmt.Sprintf("aliases=%v", m.Aliases)
		}
		est := model.EstimateVRAM(specs[id])
		fmt.Printf("%-16s %-12s %-22s %6d %4d %8d  %s\n", id, m.Runtime, strings.Join(m.Capabilities, ","), m.Ctx, m.Parallel, est.TotalMB, flags)
		if *explain {
			fmt.Printf("%-16s   %s\n", "", est)
		}
	}
	return nil
}

func cmdStatus(args []string) error {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	addr := fs.String("addr", "127.0.0.1:8080", "gridcore address")
	asJSON := fs.Bool("json", false, "print raw JSON")
	watch := fs.Bool("watch", false, "refresh every second")
	interval := fs.Duration("interval", time.Second, "refresh interval with --watch")
	_ = fs.Parse(args)

	for {
		st, raw, err := fetchState(*addr)
		if err != nil {
			return err
		}
		if *asJSON {
			os.Stdout.Write(raw)
			if !*watch {
				return nil
			}
		} else {
			if *watch {
				fmt.Print("\033[H\033[2J") // clear screen
			}
			printState(st)
			if !*watch {
				return nil
			}
		}
		time.Sleep(*interval)
	}
}

func fetchState(addr string) (scheduler.State, []byte, error) {
	resp, err := http.Get("http://" + addr + "/admin/state")
	if err != nil {
		return scheduler.State{}, nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return scheduler.State{}, nil, err
	}
	var st scheduler.State
	if err := json.Unmarshal(raw, &st); err != nil {
		return scheduler.State{}, nil, err
	}
	return st, raw, nil
}

// printState renders the terminal dashboard.
func printState(st scheduler.State) {
	g := st.GPU
	fmt.Printf("%s  mode=%s\n", st.Now.Local().Format("15:04:05"), st.Mode)
	if g.TotalMB > 0 {
		fmt.Printf("GPU %-26s %s %5.1f / %.1f GB  committed %.1f  util %d%%\n",
			g.Name, bar(g.CommittedMB, g.BudgetMB, 20), float64(g.UsedMB)/1024, float64(g.TotalMB)/1024,
			float64(g.CommittedMB)/1024, g.UtilPct)
		if g.ExternalMB > 0 {
			fmt.Printf("    external (not managed): %.1f GB\n", float64(g.ExternalMB)/1024)
		}
	} else {
		fmt.Println("GPU  (no snapshot yet)")
	}
	fmt.Println()
	fmt.Println("RUNNING")
	for _, j := range st.Running {
		fmt.Printf("  %-11s %-16s %-10s steps %d/%d  queued %dms\n", j.Class, j.Model, j.Kind, j.Completed, j.Steps, j.WaitedMS)
	}
	fmt.Println("QUEUED")
	for _, j := range st.Queued {
		fmt.Printf("  %-11s %-16s %-10s waiting %dms  %s\n", j.Class, j.Model, j.Kind, j.WaitedMS, j.Reason)
	}
	fmt.Println("RESIDENT")
	for _, r := range st.Resident {
		meas := "~"
		if r.Measured {
			meas = " "
		}
		fmt.Printf("  %-16s %-8s %-6s %s%5.1f GB  slots %d/%d\n", r.ID, r.State, r.Tier, meas, float64(r.VRAMMB)/1024, r.BusySlots, r.Slots)
	}
	if len(st.Disabled) > 0 {
		fmt.Printf("DISABLED  %s\n", strings.Join(st.Disabled, ", "))
	}
	if n := len(st.Events); n > 0 {
		fmt.Println("EVENTS")
		if n > 15 {
			st.Events = st.Events[n-15:]
		}
		for _, e := range st.Events {
			fmt.Printf("  %s %-9s %-16s %s\n", e.At.Local().Format("15:04:05.000"), e.Kind, e.Subject, e.Detail)
		}
	}
}

func bar(used, total, width int) string {
	if total <= 0 {
		return strings.Repeat(".", width)
	}
	filled := used * width / total
	if filled > width {
		filled = width
	}
	if filled < 0 {
		filled = 0
	}
	return "[" + strings.Repeat("#", filled) + strings.Repeat(".", width-filled) + "]"
}

func setupLogging(level string) {
	var lvl slog.Level
	switch level {
	case "debug":
		lvl = slog.LevelDebug
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lvl})))
}
