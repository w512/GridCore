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
	kind := flag.String("kind", "embed", "embed: embed documents in chunks; chat: classify each document with a chat model")
	model := flag.String("model", "", "model (default: first configured embedding or chat model)")
	docs := flag.Int("docs", 3000, "documents per pass")
	chunk := flag.Int("chunk", 0, "documents per progress step (default 32 for embed, 4 for chat)")
	docWords := flag.Int("doc-words", 300, "words per synthetic document; must fit the embedding model's per-slot context (nomic: ctx/parallel = 512 tokens, so keep it under ~350)")
	rate := flag.Float64("rate", 0, "cap throughput at N documents/s (0 = as fast as the GPU allows)")
	loop := flag.Bool("loop", false, "start over when a pass completes")
	class := flag.String("class", "background", "workload class")
	pause := flag.Duration("pause", 0, "extra pause between steps")
	flag.Parse()
	if *kind != "embed" && *kind != "chat" {
		fmt.Fprintln(os.Stderr, "demo-indexer: -kind must be embed or chat")
		os.Exit(2)
	}
	if *chunk == 0 {
		*chunk = 32
		if *kind == "chat" {
			*chunk = 4
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if *model == "" {
		cap := "embedding"
		if *kind == "chat" {
			cap = "chat"
		}
		m, err := firstModelWith(*addr, cap)
		if err != nil {
			fmt.Fprintln(os.Stderr, "demo-indexer:", err)
			os.Exit(1)
		}
		*model = m
	}
	client := &http.Client{}
	cfg := passConfig{addr: *addr, kind: *kind, model: *model, class: *class, docs: *docs, chunk: *chunk,
		docWords: *docWords, rate: *rate, pause: *pause}

	what := "embedding"
	if *kind == "chat" {
		what = "classifying"
	}
	fmt.Printf("%sindexer%s  %s %d docs (~%d words each) with %s  class=%s  steps of %d\n\n", bold, reset, what, *docs, *docWords, *model, *class, *chunk)
	pass := 1
	for {
		if err := runPass(ctx, client, cfg, pass); err != nil {
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

type passConfig struct {
	addr, kind, model, class string
	docs, chunk, docWords    int
	rate                     float64
	pause                    time.Duration
}

func runPass(ctx context.Context, client *http.Client, cfg passConfig, pass int) error {
	docs, chunk, addr, model, class, pause := cfg.docs, cfg.chunk, cfg.addr, cfg.model, cfg.class, cfg.pause
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
		rate := 0.0 // wall-clock, so scheduler pauses visibly pull it down
		if elapsed > 0 {
			rate = float64(docsDone) / elapsed.Seconds()
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
		batch := makeDocs(pass, lo, hi, cfg.docWords)

		// Fire the step and animate while it is queued.
		type result struct {
			queue int64
			dur   time.Duration
			err   error
		}
		resCh := make(chan result, 1)
		reqStart := time.Now()
		go func() {
			var q int64
			var err error
			if cfg.kind == "chat" {
				// One classification request per document; queue time is the
				// sum of what each request waited.
				for _, doc := range batch {
					var qq int64
					qq, err = post(ctx, client, addr, "/v1/chat/completions", class, map[string]any{
						"model": model, "max_tokens": 6, "temperature": 0,
						"messages": []map[string]string{{"role": "user", "content": "Classify this note with one word (topic). Note:\n\n" + doc}},
					})
					q += qq
					if err != nil {
						break
					}
				}
			} else {
				q, err = post(ctx, client, addr, "/v1/embeddings", class, map[string]any{"model": model, "input": batch})
			}
			resCh <- result{queue: q, dur: time.Since(reqStart), err: err}
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
		// Pacing: -rate caps documents per second; -pause adds a fixed gap.
		wait := pause
		if cfg.rate > 0 {
			target := time.Duration(float64(docsDone) / cfg.rate * float64(time.Second))
			if ahead := target - (time.Since(start) - pausedTotal); ahead > wait {
				wait = ahead
			}
		}
		if wait > 0 {
			select {
			case <-time.After(wait):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
	draw("done", 0)
	fmt.Println()
	return nil
}

// post sends one request and returns its queue time.
func post(ctx context.Context, client *http.Client, addr, path, class string, payload any) (int64, error) {
	body, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+addr+path, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-GridCore-Class", class)
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return 0, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncate(raw))
	}
	q, _ := strconv.ParseInt(resp.Header.Get("X-GridCore-Queue-Ms"), 10, 64)
	return q, nil
}

// vocabulary for synthetic documents; varied enough that prompt caching
// does not make later chunks artificially fast.
var words = strings.Fields(`scheduler gpu memory residency priority queue interactive background batch
model weights context cache eviction latency throughput tokens embedding vector index document paragraph
policy budget headroom reservation slot runtime process health metrics dashboard timeline event
release load unload pinned hot cold tier admission estimate measured profile driver kernel buffer
stream request response client server proxy header class deadline timeout retry breaker crash recover
notes meeting project quarterly review draft summary decision action owner date status risk plan`)

func makeDocs(pass, lo, hi, nWords int) []string {
	out := make([]string, 0, hi-lo)
	for i := lo; i < hi; i++ {
		seed := uint32(pass*1_000_003 + i*7919)
		var sb strings.Builder
		fmt.Fprintf(&sb, "Document %d (pass %d). ", i, pass)
		for w := 0; w < nWords; w++ {
			seed = seed*1664525 + 1013904223 // LCG; deterministic per document
			sb.WriteString(words[int(seed>>8)%len(words)])
			if w%13 == 12 {
				sb.WriteString(". ")
			} else {
				sb.WriteByte(' ')
			}
		}
		out = append(out, sb.String())
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
