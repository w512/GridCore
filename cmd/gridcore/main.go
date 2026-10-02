// Command gridcore is a workload scheduler for local AI computers.
//
//	gridcore serve   [--config PATH] [--state-dir DIR]   run the daemon
//	gridcore check   [--config PATH]                     validate config and files
//	gridcore status  [--addr HOST:PORT] [--watch]        show scheduler state
//	gridcore models  [--config PATH]                     list configured models
//	gridcore bench   [--all] <model-id>...               measure VRAM, load time, throughput
//	gridcore profiles [list | prune]                     show or clean up stored measurements
//	gridcore version
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/w512/gridcore/internal/api"
	"github.com/w512/gridcore/internal/config"
	"github.com/w512/gridcore/internal/metrics"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/scheduler"
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
	case "profiles":
		err = cmdProfiles(os.Args[2:])
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
  gridcore profiles [list | prune [--dry-run]] [--config PATH] [--state-dir DIR]
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
