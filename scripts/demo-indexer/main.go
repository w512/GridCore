// Command demo-indexer plays a background indexing job for demos: it embeds
// documents chunk by chunk and draws a progress bar that visibly pauses
// while the scheduler gives the GPU to interactive work.
//
//	go run ./scripts/demo-indexer -addr 127.0.0.1:8080 -docs 3000 -chunk 32
//	go run ./scripts/demo-indexer -loop            # keep indexing until Ctrl-C
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"
)

const (
	esc    = "\033["
	reset  = esc + "0m"
	dim    = esc + "2m"
	bold   = esc + "1m"
	green  = esc + "32m"
	yellow = esc + "33m"
	red    = esc + "31m"
	cyan   = esc + "36m"
)

func main() {
	addr := flag.String("addr", envOr("GRIDCORE_ADDR", "127.0.0.1:8080"), "gridcore address")
	model := flag.String("model", "", "embedding model (default: first configured)")
	docs := flag.Int("docs", 3000, "documents per pass")
	chunk := flag.Int("chunk", 32, "documents per request")
	loop := flag.Bool("loop", false, "start over when a pass completes")
	class := flag.String("class", "background", "workload class")
	pause := flag.Duration("pause", 0, "extra pause between chunks (to slow the demo down)")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if *model == "" {
		m, err := firstModelWith(*addr, "embedding")
		if err != nil {
			fmt.Fprintln(os.Stderr, "demo-indexer:", err)
			os.Exit(1)
		}
		*model = m
	}
	client := &http.Client{}

	fmt.Printf("%sindexer%s  model=%s  class=%s  %d docs in chunks of %d\n\n", bold, reset, *model, *class, *docs, *chunk)
	pass := 1
	for {
		if err := runPass(ctx, client, *addr, *model, *class, *docs, *chunk, *pause, pass); err != nil {
			if ctx.Err() != nil {
				fmt.Printf("\n%sstopped%s\n", dim, reset)
				return
			}
			fmt.Printf("\n%serror:%s %v\n", red, reset, err)
			os.Exit(1)
		}
		if !*loop {
			return
		}
		pass++
		time.Sleep(time.Second)
	}
}

func runPass(ctx context.Context, client *http.Client, addr, model, class string, docs, chunk int, pause time.Duration, pass int) error {
	total := (docs + chunk - 1) / chunk
	start := time.Now()
	var indexed, pausedTotal time.Duration
	var done int
	var docsDone int
	var lastQueue int64

	draw := func(state string, waiting time.Duration) {
		pct := float64(docsDone) / float64(docs)
		width := 34
		filled := int(pct * float64(width))
		bar := green + strings.Repeat("▇", filled) + dim + strings.Repeat("░", width-filled) + reset
		elapsed := time.Since(start)
		rate := 0.0
		if indexed > 0 {
			rate = float64(docsDone) / indexed.Seconds()
		}
		line := fmt.Sprintf("\r%s %3.0f%%  chunk %*d/%d  %5.0f docs/s  %s", bar, pct*100, len(strconv.Itoa(total)), done, total, rate, fmtDur(elapsed))
		switch state {
		case "running":
			line += fmt.Sprintf("  %s▶ running%s   ", green, reset)
		case "waiting":
			line += fmt.Sprintf("  %s⏸ waiting for GPU %s%s", yellow, fmtDur(waiting), reset)
		case "done":
			line += fmt.Sprintf("  %s✓ done%s  (paused %s total)", cyan, reset, fmtDur(pausedTotal))
		}
		fmt.Print(line + esc + "K")
	}

	for done < total {
		lo := done * chunk
		hi := min(lo+chunk, docs)
		body, _ := json.Marshal(map[string]any{"model": model, "input": makeDocs(pass, lo, hi)})

		// Fire the request and animate while it is queued.
		type result struct {
			queue int64
			dur   time.Duration
			err   error
		}
		resCh := make(chan result, 1)
		reqStart := time.Now()
		go func() {
			req, _ := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+addr+"/v1/embeddings", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-GridCore-Class", class)
			resp, err := client.Do(req)
			if err != nil {
				resCh <- result{err: err}
				return
			}
			defer resp.Body.Close()
			raw, _ := io.ReadAll(resp.Body)
			if resp.StatusCode != 200 {
				resCh <- result{err: fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncate(raw))}
				return
			}
			q, _ := strconv.ParseInt(resp.Header.Get("X-GridCore-Queue-Ms"), 10, 64)
			resCh <- result{queue: q, dur: time.Since(reqStart)}
		}()

		tick := time.NewTicker(100 * time.Millisecond)
		var res result
	wait:
		for {
			select {
			case res = <-resCh:
				break wait
			case <-tick.C:
				// Anything beyond a normal chunk (~250 ms) means we are queued.
				if w := time.Since(reqStart); w > 400*time.Millisecond {
					draw("waiting", w)
				} else {
					draw("running", 0)
				}
			case <-ctx.Done():
				tick.Stop()
				return ctx.Err()
			}
		}
		tick.Stop()
		if res.err != nil {
			return res.err
		}
		lastQueue = res.queue
		q := time.Duration(res.queue) * time.Millisecond
		pausedTotal += q
		indexed += res.dur - q
		done++
		docsDone = hi
		draw("running", 0)
		if lastQueue > 500 {
			// Leave a trace of the pause in the scrollback.
			fmt.Printf("\n%s   chunk %d waited %s while interactive work ran%s\n", dim, done, fmtDur(q), reset)
		}
		if pause > 0 {
			select {
			case <-time.After(pause):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
	draw("done", 0)
	fmt.Println()
	return nil
}

func makeDocs(pass, lo, hi int) []string {
	out := make([]string, 0, hi-lo)
	for i := lo; i < hi; i++ {
		out = append(out, fmt.Sprintf("Pass %d, document %d: notes on GPU memory residency, priority queues and local inference workloads.", pass, i))
	}
	return out
}

func firstModelWith(addr, cap string) (string, error) {
	resp, err := http.Get("http://" + addr + "/v1/models")
	if err != nil {
		return "", fmt.Errorf("gridcore not reachable at %s: %w", addr, err)
	}
	defer resp.Body.Close()
	var out struct {
		Data []struct {
			ID           string   `json:"id"`
			Root         string   `json:"root"`
			Capabilities []string `json:"capabilities"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	for _, m := range out.Data {
		if m.Root != "" {
			continue
		}
		for _, c := range m.Capabilities {
			if c == cap {
				return m.ID, nil
			}
		}
	}
	return "", fmt.Errorf("no model with capability %q configured", cap)
}

func fmtDur(d time.Duration) string {
	switch {
	case d < time.Second:
		return fmt.Sprintf("%dms", d.Milliseconds())
	case d < time.Minute:
		return fmt.Sprintf("%.1fs", d.Seconds())
	default:
		return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
	}
}

func truncate(b []byte) string {
	s := strings.TrimSpace(string(b))
	if len(s) > 200 {
		return s[:200] + "..."
	}
	return s
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
