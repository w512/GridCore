// Command gridcore is a workload scheduler for local AI computers.
//
//	gridcore serve   [--config PATH] [--state-dir DIR]   run the daemon
//	gridcore check   [--config PATH]                     validate config and files
//	gridcore status  [--addr HOST:PORT]                  show scheduler state
//	gridcore models  [--config PATH]                     list configured models
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
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/gridcore/gridcore/internal/api"
	"github.com/gridcore/gridcore/internal/config"
	"github.com/gridcore/gridcore/internal/metrics"
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
  gridcore serve   [--config PATH] [--state-dir DIR]
  gridcore check   [--config PATH]
  gridcore status  [--addr HOST:PORT]
  gridcore models  [--config PATH]
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
	if err := os.MkdirAll(cfg.StateDir, 0o755); err != nil {
		return fmt.Errorf("state dir: %w", err)
	}

	m := metrics.New()
	// TODO(M1): construct gpu monitor, runtimes, scheduler; pass scheduler as
	// StateProvider and wire inference endpoints through the proxy.
	srv := api.New(cfg, m, nil, version)

	httpSrv := &http.Server{
		Addr:              cfg.Server.Listen,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: streaming responses and long generations are normal.
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		slog.Info("gridcore listening", "addr", cfg.Server.Listen, "models", len(cfg.Models), "gpu", cfg.GPU.Device, "state_dir", cfg.StateDir)
		if err := httpSrv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
	}
	slog.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
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
	return nil
}

func cmdModels(args []string) error {
	fs := flag.NewFlagSet("models", flag.ExitOnError)
	cfgPath := configFlag(fs)
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
	fmt.Printf("%-16s %-12s %-22s %6s %4s  %s\n", "ID", "RUNTIME", "CAPABILITIES", "CTX", "PAR", "FLAGS")
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
		fmt.Printf("%-16s %-12s %-22s %6d %4d  %s\n", id, m.Runtime, strings.Join(m.Capabilities, ","), m.Ctx, m.Parallel, flags)
	}
	return nil
}

func cmdStatus(args []string) error {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	addr := fs.String("addr", "127.0.0.1:8080", "gridcore address")
	asJSON := fs.Bool("json", false, "print raw JSON")
	_ = fs.Parse(args)

	resp, err := http.Get("http://" + *addr + "/admin/state")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if *asJSON {
		os.Stdout.Write(body)
		return nil
	}
	var st api.State
	if err := json.Unmarshal(body, &st); err != nil {
		return err
	}
	printState(st)
	return nil
}

// printState renders the terminal dashboard. Until the
// scheduler exists most sections will be empty.
func printState(st api.State) {
	g := st.GPU
	if g.TotalMB > 0 {
		fmt.Printf("GPU %-28s %s %5.1f / %.1f GB  util %d%%  mode=%s\n",
			g.Name, bar(g.UsedMB, g.TotalMB, 20), float64(g.UsedMB)/1024, float64(g.TotalMB)/1024, g.UtilPct, g.Mode)
	} else {
		fmt.Println("GPU  (no data)")
	}
	fmt.Println()
	fmt.Println("RUNNING")
	for _, j := range st.Running {
		fmt.Printf("  %-10s %-16s %-12s step %d/%d\n", j.Class, j.Model, j.Kind, j.Step, j.Steps)
	}
	fmt.Println("QUEUED")
	for _, j := range st.Queued {
		fmt.Printf("  %-10s %-16s %-12s waited %s\n", j.Class, j.Model, j.Kind, j.Waited.Round(time.Millisecond))
	}
	fmt.Println("RESIDENT")
	for _, r := range st.Resident {
		fmt.Printf("  %-16s %-6s %6.1f GB  slots %d/%d  %s\n", r.ID, r.Tier, float64(r.VRAMMB)/1024, r.BusySlots, r.Slots, r.State)
	}
	if len(st.Events) > 0 {
		fmt.Println("EVENTS")
		for _, e := range st.Events {
			fmt.Printf("  %s %-9s %-16s %s\n", e.At.Format("15:04:05"), e.Kind, e.Subject, e.Detail)
		}
	}
}

func bar(used, total, width int) string {
	if total <= 0 {
		return ""
	}
	filled := used * width / total
	if filled > width {
		filled = width
	}
	b := make([]byte, 0, width)
	for i := 0; i < width; i++ {
		if i < filled {
			b = append(b, '#')
		} else {
			b = append(b, '.')
		}
	}
	return "[" + string(b) + "]"
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
