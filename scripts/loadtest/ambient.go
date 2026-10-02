package main

// The v0.2 acceptance scenario from docs/idea.md: one GPU, five kinds of
// work at once.
//
//	chat     a person chatting (interactive, streaming, long pauses)
//	agent    a coding agent (long prompts, back to back; -agent-class)
//	ambient  a screenshot analyser (background, an image every few seconds)
//	indexer  a file indexer (background embeddings, continuous)
//	ocr      a periodic OCR job (batch, an image every -ocr-every)
//
// Model flags accept models or families; the report shows which variant
// served each role, and the scheduler's loads, evictions and thrash
// warnings during the run (from /metrics).

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type ambientConfig struct {
	chat, agent, ambient, indexer, ocr int

	chatModel, agentModel, ambientModel, embedModel, ocrModel string
	agentClass                                                string

	ambientEvery, ocrEvery time.Duration
	agentContext           int // approximate prompt tokens per agent request
	imageSize              int
}

func ambientFlags() *ambientConfig {
	c := &ambientConfig{}
	flag.IntVar(&c.chat, "chat", 1, "ambient scenario: chatting users")
	flag.IntVar(&c.agent, "agent", 1, "ambient scenario: coding agents")
	flag.IntVar(&c.ambient, "ambient", 1, "ambient scenario: screenshot analysers")
	flag.IntVar(&c.indexer, "indexer", 1, "ambient scenario: file indexers")
	flag.IntVar(&c.ocr, "ocr", 1, "ambient scenario: OCR jobs")
	flag.StringVar(&c.chatModel, "chat-model", "gemma4", "model or family for chat")
	flag.StringVar(&c.agentModel, "agent-model", "gemma4", "model or family for the coding agent")
	flag.StringVar(&c.ambientModel, "ambient-model", "gemma4", "vision model or family for screenshots")
	flag.StringVar(&c.embedModel, "embed-model", "nomic-embed", "embedding model for the indexer")
	flag.StringVar(&c.ocrModel, "ocr-model", "gemma4", "vision model or family for OCR")
	flag.StringVar(&c.agentClass, "agent-class", "interactive", "class of the coding agent's requests")
	flag.DurationVar(&c.ambientEvery, "ambient-every", 5*time.Second, "one screenshot per analyser this often")
	flag.DurationVar(&c.ocrEvery, "ocr-every", 20*time.Second, "one OCR image per job this often")
	flag.IntVar(&c.agentContext, "agent-ctx", 3000, "approximate prompt tokens per agent request")
	flag.IntVar(&c.imageSize, "image", 512, "screenshot width and height in pixels")
	return c
}

func runAmbient(ctx context.Context, client *http.Client, addr string, c *ambientConfig) []sample {
	var (
		mu      sync.Mutex
		samples []sample
		wg      sync.WaitGroup
	)
	record := func(s sample) {
		mu.Lock()
		samples = append(samples, s)
		mu.Unlock()
	}
	start := func(n int, role string, loop func(rng *rand.Rand) (sample, time.Duration)) {
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func(seed int64) {
				defer wg.Done()
				rng := rand.New(rand.NewSource(seed))
				for ctx.Err() == nil {
					s, pause := loop(rng)
					s.class = role
					if ctx.Err() != nil && s.status == 0 {
						s.status = statusUnfinished // cut off by the end of the run
					}
					record(s)
					if ctx.Err() != nil {
						return
					}
					select {
					case <-time.After(pause):
					case <-ctx.Done():
					}
				}
			}(int64(len(role)*1000 + i))
		}
	}
	jitter := func(rng *rand.Rand, d time.Duration) time.Duration {
		return time.Duration(float64(d) * (0.5 + rng.Float64()))
	}
	// Fixed-rate roles: the next request is due `every` after this one
	// started, however long it took.
	paced := func(every time.Duration, f func(rng *rand.Rand) sample) func(rng *rand.Rand) (sample, time.Duration) {
		return func(rng *rand.Rand) (sample, time.Duration) {
			t0 := time.Now()
			s := f(rng)
			return s, max(0, every-time.Since(t0))
		}
	}

	start(c.chat, "chat", func(rng *rand.Rand) (sample, time.Duration) {
		msgs := []any{map[string]any{"role": "user", "content": fmt.Sprintf("In three sentences, why would someone pick number %d for a lottery ticket?", rng.Intn(100))}}
		return chat(ctx, client, addr, c.chatModel, "interactive", msgs, 128, true), jitter(rng, 12*time.Second)
	})
	start(c.agent, "agent", func(rng *rand.Rand) (sample, time.Duration) {
		msgs := []any{
			map[string]any{"role": "system", "content": "You are a coding agent. Answer with a unified diff only."},
			map[string]any{"role": "user", "content": codeContext(rng, c.agentContext) + "\n\nRename the function handle to process everywhere."},
		}
		return chat(ctx, client, addr, c.agentModel, c.agentClass, msgs, 256, false), jitter(rng, 2*time.Second)
	})
	start(c.ambient, "ambient", paced(c.ambientEvery, func(rng *rand.Rand) sample {
		return chat(ctx, client, addr, c.ambientModel, "background", imageMessage(rng, c.imageSize, "Describe this screenshot in one sentence."), 48, false)
	}))
	start(c.ocr, "ocr", paced(c.ocrEvery, func(rng *rand.Rand) sample {
		return chat(ctx, client, addr, c.ocrModel, "batch", imageMessage(rng, c.imageSize, "Transcribe all text in this image."), 96, false)
	}))
	start(c.indexer, "indexer", func(rng *rand.Rand) (sample, time.Duration) {
		return embedDocs(ctx, client, addr, c.embedModel, rng, 32), jitter(rng, 500*time.Millisecond)
	})

	go func() {
		tick := time.NewTicker(10 * time.Second)
		defer tick.Stop()
		t0 := time.Now()
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
				mu.Lock()
				n := len(samples)
				mu.Unlock()
				fmt.Fprintf(os.Stderr, "  %4.0fs  %d requests\n", time.Since(t0).Seconds(), n)
			}
		}
	}()
	wg.Wait()
	return samples
}

func chat(ctx context.Context, c *http.Client, addr, model, class string, msgs []any, maxTokens int, stream bool) sample {
	body, _ := json.Marshal(map[string]any{"model": model, "messages": msgs, "max_tokens": maxTokens, "stream": stream})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+addr+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-GridCore-Class", class)
	return do(c, req, class)
}

func embedDocs(ctx context.Context, c *http.Client, addr, model string, rng *rand.Rand, n int) sample {
	docs := make([]string, n)
	for i := range docs {
		docs[i] = prose(rng, 60+rng.Intn(120))
	}
	body, _ := json.Marshal(map[string]any{"model": model, "input": docs})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+addr+"/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-GridCore-Class", "background")
	return do(c, req, "background")
}

var words = strings.Fields("the scheduler keeps models resident while they are useful and evicts the ones that are cheapest to lose when a request needs room on the card budget interactive background batch queue slot memory weights cache context token prompt answer file index screenshot window editor terminal")

func prose(rng *rand.Rand, n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(words[rng.Intn(len(words))])
	}
	return b.String()
}

// codeContext is roughly tokens tokens of Go-looking source.
func codeContext(rng *rand.Rand, tokens int) string {
	var b strings.Builder
	for i := 0; b.Len() < tokens*4; i++ {
		fmt.Fprintf(&b, "func handle%d(ctx context.Context, req *Request) (*Response, error) {\n\tif req.N > %d {\n\t\treturn nil, errTooMany\n\t}\n\treturn &Response{ID: req.ID, Sum: req.N * %d}, nil\n}\n\n",
			i, rng.Intn(1000), rng.Intn(100))
	}
	return b.String()
}

// imageMessage is a user message with a synthetic "screenshot": windows,
// a title bar and text-like strokes, different every time so nothing is
// served from a cache.
func imageMessage(rng *rand.Rand, size int, prompt string) []any {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	fill := func(x0, y0, x1, y1 int, c color.RGBA) {
		for y := max(0, y0); y < min(size, y1); y++ {
			for x := max(0, x0); x < min(size, x1); x++ {
				img.Set(x, y, c)
			}
		}
	}
	fill(0, 0, size, size, color.RGBA{230, 232, 236, 255})
	for w := 0; w < 3; w++ {
		x, y := rng.Intn(size/2), rng.Intn(size/2)
		ww, hh := size/4+rng.Intn(size/3), size/4+rng.Intn(size/3)
		fill(x, y, x+ww, y+hh, color.RGBA{255, 255, 255, 255})
		fill(x, y, x+ww, y+14, color.RGBA{uint8(rng.Intn(120)), uint8(80 + rng.Intn(120)), 200, 255})
		for line := y + 22; line < y+hh-8; line += 12 {
			fill(x+8, line, x+8+rng.Intn(max(1, ww-16)), line+5, color.RGBA{40, 40, 40, 255})
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	url := "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
	return []any{map[string]any{"role": "user", "content": []any{
		map[string]any{"type": "text", "text": prompt},
		map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}},
	}}}
}

// reportModels shows which model served each role (families pick one).
func reportModels(samples []sample) {
	by := map[string]map[string]int{}
	for _, s := range samples {
		if s.status != 200 {
			continue
		}
		if by[s.class] == nil {
			by[s.class] = map[string]int{}
		}
		by[s.class][s.model]++
	}
	roles := make([]string, 0, len(by))
	for r := range by {
		roles = append(roles, r)
	}
	sort.Strings(roles)
	fmt.Printf("\n%-12s %s\n", "ROLE", "SERVED BY")
	for _, r := range roles {
		var parts []string
		for m, n := range by[r] {
			parts = append(parts, fmt.Sprintf("%s %d", m, n))
		}
		sort.Strings(parts)
		fmt.Printf("%-12s %s\n", r, strings.Join(parts, ", "))
	}
}

// scrapeCounters reads the scheduler counters the scenario is judged by.
func scrapeCounters(addr string) map[string]float64 {
	out := map[string]float64{}
	resp, err := http.Get("http://" + addr + "/metrics")
	if err != nil {
		return out
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "gridcore_model_loads_total") && !strings.HasPrefix(line, "gridcore_model_evictions_total") &&
			!strings.HasPrefix(line, "gridcore_model_thrash_total") {
			continue
		}
		i := strings.LastIndexByte(line, ' ')
		if v, err := strconv.ParseFloat(line[i+1:], 64); err == nil {
			out[line[:i]] = v
		}
	}
	return out
}

func reportCounters(before, after map[string]float64) {
	var lines []string
	for k, v := range after {
		if d := v - before[k]; d > 0 {
			lines = append(lines, fmt.Sprintf("  %-70s %4.0f", k, d))
		}
	}
	sort.Strings(lines)
	fmt.Println("\nSCHEDULER DURING THE RUN (loads, evictions, thrash)")
	if len(lines) == 0 {
		fmt.Println("  none")
	}
	for _, l := range lines {
		fmt.Println(l)
	}
}
