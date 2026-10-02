package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/w512/gridcore/internal/scheduler"
)

// ANSI styling for the dashboard. Disabled with --no-color, NO_COLOR, or
// when stdout is not a terminal.
type palette struct {
	reset, bold, dim, red, green, yellow, blue, magenta, cyan string
}

var mono = palette{}

var ansi = palette{
	reset: "\033[0m", bold: "\033[1m", dim: "\033[2m",
	red: "\033[31m", green: "\033[32m", yellow: "\033[33m", blue: "\033[34m", magenta: "\033[35m", cyan: "\033[36m",
}

func cmdStatus(args []string) error {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	addr := fs.String("addr", envOr("GRIDCORE_ADDR", "127.0.0.1:8080"), "gridcore address")
	asJSON := fs.Bool("json", false, "print raw JSON")
	watch := fs.Bool("watch", false, "refresh continuously")
	interval := fs.Duration("interval", 500*time.Millisecond, "refresh interval with --watch")
	noColor := fs.Bool("no-color", false, "disable colors")
	events := fs.Int("events", 12, "number of recent events to show")
	_ = fs.Parse(args)

	pal := ansi
	if *noColor || os.Getenv("NO_COLOR") != "" || !isTerminal(os.Stdout) {
		pal = mono
	}
	if *watch && pal != mono {
		fmt.Print("\033[?25l")       // hide cursor
		defer fmt.Print("\033[?25h") // show cursor
	}

	var prev string
	for {
		st, raw, err := fetchState(*addr)
		if err != nil {
			if !*watch {
				return err
			}
			fmt.Printf("\033[H\033[2J%sgridcore at %s not reachable:%s %v\n", pal.red, *addr, pal.reset, err)
			time.Sleep(*interval)
			continue
		}
		if *asJSON {
			os.Stdout.Write(raw)
			if !*watch {
				return nil
			}
		} else {
			out := renderState(st, pal, *events)
			if !*watch {
				fmt.Print(out)
				return nil
			}
			if out != prev {
				// Home + redraw with line clears; no full-screen clear, so no flicker.
				var b strings.Builder
				b.WriteString("\033[H")
				for _, line := range strings.Split(out, "\n") {
					b.WriteString(line)
					b.WriteString("\033[K\n")
				}
				b.WriteString("\033[J")
				fmt.Print(b.String())
				prev = out
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

// renderState draws the dashboard.
func renderState(st scheduler.State, p palette, maxEvents int) string {
	var b strings.Builder
	w := func(format string, a ...any) { fmt.Fprintf(&b, format, a...) }
	gb := func(mb int) string { return fmt.Sprintf("%.1f", float64(mb)/1024) }

	// Header
	mode := st.Mode
	modeCol := p.green
	if mode == "interactive" {
		modeCol = p.magenta
	}
	w("%sgridcore%s  %s  %s%s%s\n", p.bold, p.reset, st.Now.Local().Format("15:04:05"), modeCol, strings.ToUpper(mode), p.reset)

	// GPU line
	g := st.GPU
	if g.TotalMB > 0 {
		used := g.CommittedMB
		pressure := false
		for _, j := range st.Queued {
			if strings.Contains(j.Reason, "VRAM") || strings.Contains(j.Reason, "evict") {
				pressure = true
			}
		}
		w("%s%-27s%s %s %s / %s GB", p.bold, trunc(g.Name, 27), p.reset, gaugeBar(used, g.BudgetMB, 30, pressure, p), gb(used), gb(g.BudgetMB))
		if g.UtilPct >= 0 {
			w("  util %3d%%", g.UtilPct)
		}
		w("\n")
		if g.ExternalMB > 64 {
			w("%s%-27s external (not managed) %s GB%s\n", p.dim, "", gb(g.ExternalMB), p.reset)
		}
		if g.MemoryKind == "unified" {
			col := p.dim
			switch g.Pressure {
			case "warn":
				col = p.yellow
			case "critical":
				col = p.red
			}
			w("%-27s %sunified memory · pressure %s%s%s · swap %s GB%s\n", "", p.dim, col, g.Pressure, p.dim, gb(g.SwapUsedMB), p.reset)
		}
	} else {
		w("%sGPU: no snapshot yet%s\n", p.dim, p.reset)
	}
	w("\n")

	// Resident models: one bar each, proportional to the budget.
	w("%sRESIDENT%s\n", p.bold, p.reset)
	if len(st.Resident) == 0 {
		w("%s  (nothing loaded)%s\n", p.dim, p.reset)
	}
	for _, r := range st.Resident {
		tierCol := p.dim
		switch r.Tier {
		case "hot":
			tierCol = p.magenta
		case "pinned":
			tierCol = p.blue
		}
		stateStr := r.State
		stateCol := p.green
		switch r.State {
		case "loading":
			stateCol = p.yellow
		case "draining", "stopping":
			stateCol = p.red
		}
		slots := fmt.Sprintf("%d/%d", r.BusySlots, r.Slots)
		if r.BusySlots > 0 {
			slots = p.green + p.bold + slots + p.reset
		} else {
			slots = p.dim + slots + p.reset
		}
		meas := ""
		if !r.Measured {
			meas = "~"
		}
		// What losing the model would cost (reload seconds x recent demand):
		// the cheapest allowed models are evicted first.
		cost := ""
		if r.Tier != "pinned" && r.EvictCost >= 0.05 {
			cost = fmt.Sprintf("  %scost %.1f%s", p.dim, r.EvictCost, p.reset)
		}
		w("  %-16s %s%-8s%s %s%-6s%s %s %s%5s GB  slots %s%s\n",
			trunc(r.ID, 16), stateCol, stateStr, p.reset, tierCol, r.Tier, p.reset,
			modelBar(r.VRAMMB, g.BudgetMB, 20, tierCol, p), meas, gb(r.VRAMMB), slots, cost)
	}
	if len(st.Disabled) > 0 {
		w("  %sDISABLED  %s%s\n", p.red, strings.Join(st.Disabled, ", "), p.reset)
	}
	w("\n")

	// Jobs
	w("%sRUNNING%s", p.bold, p.reset)
	if len(st.Running) == 0 {
		w("  %s—%s", p.dim, p.reset)
	}
	w("\n")
	for _, j := range st.Running {
		w("  %s%-11s%s %-16s %-10s %s\n", classColor(j.Class, p), j.Class, p.reset, trunc(j.Model, 16), j.Kind, stepsStr(j, p))
	}
	w("%sQUEUED%s", p.bold, p.reset)
	if len(st.Queued) == 0 {
		w("  %s—%s", p.dim, p.reset)
	}
	w("\n")
	for _, j := range st.Queued {
		w("  %s%-11s%s %-16s %-10s waiting %s%s%s  %s%s%s\n", classColor(j.Class, p), j.Class, p.reset, trunc(j.Model, 16), j.Kind,
			p.yellow, fmtMS(j.WaitedMS), p.reset, p.dim, j.Reason, p.reset)
	}
	w("\n")

	// Events, newest last, with time relative to now.
	w("%sEVENTS%s\n", p.bold, p.reset)
	ev := st.Events
	if len(ev) > maxEvents {
		ev = ev[len(ev)-maxEvents:]
	}
	for _, e := range ev {
		age := st.Now.Sub(e.At)
		w("  %s%7s%s  %s%-9s%s %-16s %s%s%s\n", p.dim, "-"+fmtAge(age), p.reset, eventColor(e.Kind, p), e.Kind, p.reset,
			shortSubject(e.Subject), p.dim, trunc(e.Detail, 60), p.reset)
	}
	return b.String()
}

func stepsStr(j scheduler.JobState, p palette) string {
	if j.Steps == 1 {
		return fmt.Sprintf("%s%s%s", p.green, "▶ generating", p.reset)
	}
	return fmt.Sprintf("%s▶%s step %d/%d", p.green, p.reset, j.Completed+1, j.Steps)
}

// gaugeBar is the GPU gauge. A full card is what a residency scheduler
// aims for, so fullness alone is not alarming: the bar turns red only when
// something is queued waiting for VRAM.
func gaugeBar(used, total, width int, pressure bool, p palette) string {
	if total <= 0 {
		return "[" + strings.Repeat("·", width) + "]"
	}
	filled := used * width / total
	filled = max(0, min(filled, width))
	col := p.cyan
	if pressure {
		col = p.red
	}
	return "[" + col + strings.Repeat("█", filled) + p.reset + p.dim + strings.Repeat("·", width-filled) + p.reset + "]"
}

// modelBar shows one model's share of the budget.
func modelBar(mb, total, width int, col string, p palette) string {
	if total <= 0 {
		return strings.Repeat(" ", width)
	}
	filled := mb * width / total
	filled = max(1, min(filled, width))
	return col + strings.Repeat("▇", filled) + p.reset + p.dim + strings.Repeat("░", width-filled) + p.reset
}

func classColor(c string, p palette) string {
	switch c {
	case "interactive":
		return p.magenta
	case "background":
		return p.cyan
	default:
		return p.blue
	}
}

func eventColor(kind string, p palette) string {
	switch kind {
	case "evict", "unloaded", "crash", "load_fail", "disabled", "fail", "timeout":
		return p.red
	case "load", "loaded", "enabled":
		return p.yellow
	case "dispatch", "complete":
		return p.green
	case "mode", "preempt":
		return p.magenta
	default:
		return ""
	}
}

func fmtMS(ms int64) string {
	if ms < 1000 {
		return fmt.Sprintf("%dms", ms)
	}
	return fmt.Sprintf("%.1fs", float64(ms)/1000)
}

func fmtAge(d time.Duration) string {
	switch {
	case d < 0:
		return "0.0s"
	case d < 10*time.Second:
		return fmt.Sprintf("%.1fs", d.Seconds())
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
	default:
		return fmt.Sprintf("%dh%02dm", int(d.Hours()), int(d.Minutes())%60)
	}
}

// shortSubject abbreviates 16-hex job ids to 8 characters; model names and
// "gpu" pass through.
func shortSubject(s string) string {
	if len(s) == 16 && strings.Trim(s, "0123456789abcdef") == "" {
		return s[:8]
	}
	return trunc(s, 16)
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "…"
}

func isTerminal(f *os.File) bool {
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
