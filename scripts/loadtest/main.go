// Command loadtest hammers a GridCore instance with a mix of workload
// classes and models and reports latency percentiles per class.
//
//	go run ./scripts/loadtest -addr 127.0.0.1:8080 -duration 5m -clients 20 \
//	    -interactive gemma4-12b,ornith -background ambient -batch tiny -embed nomic-embed
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type sample struct {
	class   string
	latency time.Duration
	queue   int64
	status  int
	err     error
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8080", "gridcore address")
	dur := flag.Duration("duration", 2*time.Minute, "test duration")
	clients := flag.Int("clients", 20, "concurrent clients")
	interactive := flag.String("interactive", "gemma4-12b", "comma-separated chat models for interactive traffic")
	background := flag.String("background", "gemma4-e4b", "chat models for background traffic")
	batch := flag.String("batch", "gemma4-e2b", "chat models for batch traffic")
	embed := flag.String("embed", "nomic-embed", "embedding model for background embeddings ('' to disable)")
	maxTokens := flag.Int("max-tokens", 32, "max_tokens per chat request")
	thinkTime := flag.Duration("think", 500*time.Millisecond, "average pause between requests per client")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), *dur)
	defer cancel()
	client := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: *clients * 2}}

	var (
		mu      sync.Mutex
		samples []sample
		total   atomic.Int64
	)
	record := func(s sample) {
		mu.Lock()
		samples = append(samples, s)
		mu.Unlock()
		total.Add(1)
	}

	// Client roles: 30% interactive, 40% background, 20% batch, 10% embeddings.
	var wg sync.WaitGroup
	for i := 0; i < *clients; i++ {
		role := "interactive"
		switch {
		case i%10 == 9 && *embed != "":
			role = "embed"
		case i%10 >= 7:
			role = "batch"
		case i%10 >= 3:
			role = "background"
		}
		wg.Add(1)
		go func(id int, role string) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(id)))
			for ctx.Err() == nil {
				var s sample
				switch role {
				case "embed":
					s = embedReq(ctx, client, *addr, *embed, 20+rng.Intn(60))
				default:
					models := strings.Split(map[string]string{"interactive": *interactive, "background": *background, "batch": *batch}[role], ",")
					model := strings.TrimSpace(models[rng.Intn(len(models))])
					s = chatReq(ctx, client, *addr, model, role, *maxTokens, rng.Intn(2) == 0)
				}
				if ctx.Err() != nil {
					break
				}
				record(s)
				pause := time.Duration(float64(*thinkTime) * (0.5 + rng.Float64()))
				if role == "interactive" {
					pause *= 3 // humans type slower than indexers
				}
				select {
				case <-time.After(pause):
				case <-ctx.Done():
				}
			}
		}(i, role)
	}

	// Progress line.
	go func() {
		tick := time.NewTicker(10 * time.Second)
		defer tick.Stop()
		start := time.Now()
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
				fmt.Fprintf(os.Stderr, "  %4.0fs  %d requests\n", time.Since(start).Seconds(), total.Load())
			}
		}
	}()
	wg.Wait()

	report(samples)
}

func chatReq(ctx context.Context, c *http.Client, addr, model, class string, maxTokens int, stream bool) sample {
	body, _ := json.Marshal(map[string]any{
		"model":      model,
		"messages":   []map[string]string{{"role": "user", "content": fmt.Sprintf("Give me a %d-word fact about the number %d.", 5+rand.Intn(10), rand.Intn(10000))}},
		"max_tokens": maxTokens,
		"stream":     stream,
	})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+addr+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-GridCore-Class", class)
	return do(c, req, class)
}

func embedReq(ctx context.Context, c *http.Client, addr, model string, n int) sample {
	docs := make([]string, n)
	for i := range docs {
		docs[i] = fmt.Sprintf("Document %d about GPU scheduling, model residency and priority queues.", rand.Intn(100000))
	}
	body, _ := json.Marshal(map[string]any{"model": model, "input": docs})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+addr+"/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-GridCore-Class", "background")
	return do(c, req, "embed")
}

func do(c *http.Client, req *http.Request, class string) sample {
	start := time.Now()
	resp, err := c.Do(req)
	s := sample{class: class}
	if err != nil {
		s.err = err
		s.latency = time.Since(start)
		return s
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	s.latency = time.Since(start)
	s.status = resp.StatusCode
	fmt.Sscanf(resp.Header.Get("X-GridCore-Queue-Ms"), "%d", &s.queue)
	return s
}

func report(samples []sample) {
	byClass := map[string][]sample{}
	for _, s := range samples {
		byClass[s.class] = append(byClass[s.class], s)
	}
	classes := make([]string, 0, len(byClass))
	for c := range byClass {
		classes = append(classes, c)
	}
	sort.Strings(classes)
	fmt.Printf("\n%-12s %6s %6s %6s  %8s %8s %8s  %8s %8s\n", "CLASS", "N", "OK", "ERR", "p50", "p95", "p99", "queue50", "queue95")
	for _, c := range classes {
		ss := byClass[c]
		var lat []time.Duration
		var q []int64
		ok, errs := 0, 0
		codes := map[int]int{}
		for _, s := range ss {
			if s.err != nil {
				errs++
				continue
			}
			codes[s.status]++
			if s.status == 200 {
				ok++
				lat = append(lat, s.latency)
				q = append(q, s.queue)
			} else {
				errs++
			}
		}
		sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
		sort.Slice(q, func(i, j int) bool { return q[i] < q[j] })
		pct := func(p float64) time.Duration {
			if len(lat) == 0 {
				return 0
			}
			return lat[int(float64(len(lat)-1)*p)]
		}
		qpct := func(p float64) int64 {
			if len(q) == 0 {
				return 0
			}
			return q[int(float64(len(q)-1)*p)]
		}
		fmt.Printf("%-12s %6d %6d %6d  %8s %8s %8s  %6dms %6dms", c, len(ss), ok, errs,
			pct(0.5).Round(time.Millisecond), pct(0.95).Round(time.Millisecond), pct(0.99).Round(time.Millisecond), qpct(0.5), qpct(0.95))
		if len(codes) > 1 || codes[200] == 0 {
			fmt.Printf("  codes=%v", codes)
		}
		fmt.Println()
	}
}
