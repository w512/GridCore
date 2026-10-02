package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/w512/gridcore/internal/config"
	"github.com/w512/gridcore/internal/gpu"
	"github.com/w512/gridcore/internal/model"
	"github.com/w512/gridcore/internal/runtime"
	"github.com/w512/gridcore/internal/scheduler"
)

// benchResult is one measured model.
type benchResult struct {
	Model     string
	EstMB     int
	VRAMMB    int
	LoadMS    int64
	PromptTPS float64
	GenTPS    float64
	Err       error
}

// cmdBench loads each requested model directly through its runtime (the
// scheduler is not involved), measures VRAM, load time and throughput, and
// records the results in profiles.json so the scheduler admits the model
// with real numbers from the first request.
func cmdBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	cfgPath := configFlag(fs)
	stateDir := fs.String("state-dir", "", "state directory (default $XDG_STATE_HOME/gridcore)")
	all := fs.Bool("all", false, "bench every configured model")
	promptTok := fs.Int("prompt", 512, "approximate prompt length in tokens")
	genTok := fs.Int("gen", 128, "tokens to generate")
	noSave := fs.Bool("no-save", false, "do not update profiles.json")
	_ = fs.Parse(args)

	cfg, err := loadConfig(*cfgPath, *stateDir)
	if err != nil {
		return err
	}
	if err := cfg.CheckFiles(); err != nil {
		return err
	}
	ids := fs.Args()
	if *all {
		ids = ids[:0]
		for id := range cfg.Models {
			ids = append(ids, id)
		}
		sort.Strings(ids)
	}
	if len(ids) == 0 {
		return fmt.Errorf("usage: gridcore bench [--all] <model-id>...")
	}
	for _, id := range ids {
		if _, ok := cfg.Models[id]; !ok {
			return fmt.Errorf("unknown model %q", id)
		}
	}

	// The daemon may be running; make sure it is not holding the GPU we are
	// about to measure against (its instances would skew "before" readings
	// only, not per-PID attribution, so this is a warning, not an error).
	mon, runtimes, err := buildBackends(cfg, "")
	if err != nil {
		return err
	}
	ctx := context.Background()
	snap, err := mon.Snapshot(ctx)
	if err != nil {
		return fmt.Errorf("gpu snapshot: %w", err)
	}
	fmt.Printf("GPU: %s, %d MB total, %d MB in use before bench\n\n", snap.Name, snap.TotalMB, snap.UsedMB)

	var store *model.Store
	if !*noSave {
		store, err = model.OpenStore(filepath.Join(cfg.StateDir, "profiles.json"))
		if err != nil {
			return err
		}
	}
	specs := model.FromConfig(cfg)

	var results []benchResult
	for _, id := range ids {
		sp := specs[id]
		fmt.Printf("== %s\n", id)
		r := benchOne(ctx, cfg, runtimes[sp.Runtime], mon, sp, *promptTok, *genTok)
		results = append(results, r)
		if r.Err != nil {
			fmt.Printf("   error: %v\n\n", r.Err)
			continue
		}
		fmt.Printf("   vram %d MB (estimate was %d)  load %d ms  prompt %.0f tok/s  gen %.1f tok/s\n\n",
			r.VRAMMB, r.EstMB, r.LoadMS, r.PromptTPS, r.GenTPS)
		if store != nil {
			rtID := scheduler.RuntimeID(cfg.Runtimes[sp.Runtime], runtimes[sp.Runtime])
			_ = store.Update(sp.ProfileKey(rtID, snap.Name), id, func(p *model.Profile) {
				p.Runtime, p.GPU = rtID, snap.Name
				p.ObserveVRAM(r.VRAMMB)
				p.ObserveLoad(time.Duration(r.LoadMS) * time.Millisecond)
				p.ObserveThroughput(r.PromptTPS, r.GenTPS)
			})
		}
	}

	fmt.Printf("%-16s %9s %9s %8s %10s %9s\n", "MODEL", "VRAM MB", "EST MB", "LOAD ms", "PROMPT t/s", "GEN t/s")
	for _, r := range results {
		if r.Err != nil {
			fmt.Printf("%-16s %s\n", r.Model, "error")
			continue
		}
		fmt.Printf("%-16s %9d %9d %8d %10.0f %9.1f\n", r.Model, r.VRAMMB, r.EstMB, r.LoadMS, r.PromptTPS, r.GenTPS)
	}
	if store != nil {
		fmt.Printf("\nprofiles updated: %s\n", filepath.Join(cfg.StateDir, "profiles.json"))
	}
	return nil
}

func benchOne(ctx context.Context, cfg *config.Config, rt runtime.Runtime, mon gpu.Monitor, sp *model.Spec, promptTok, genTok int) benchResult {
	res := benchResult{Model: sp.ID, EstMB: sp.EstimateVRAMMB()}
	port, err := pickFreePort(cfg.Runtimes[sp.Runtime].PortRange)
	if err != nil {
		res.Err = err
		return res
	}
	start := time.Now()
	inst, err := rt.Load(ctx, sp, port)
	if err != nil {
		res.Err = fmt.Errorf("load: %w", err)
		return res
	}
	res.LoadMS = time.Since(start).Milliseconds()
	defer func() { _ = inst.Stop(context.Background()) }()

	// Warm-up: the first request allocates compute buffers; VRAM must be
	// read after it.
	if sp.HasCapability(config.CapChat) {
		if _, err := benchChat(ctx, inst.Addr(), sp.ID, promptTok, genTok); err != nil {
			res.Err = fmt.Errorf("warm-up: %w", err)
			return res
		}
		u, err := benchChat(ctx, inst.Addr(), sp.ID, promptTok, genTok)
		if err != nil {
			res.Err = err
			return res
		}
		res.PromptTPS, res.GenTPS = u.prompt, u.gen
	} else if sp.HasCapability(config.CapEmbedding) {
		if _, err := benchEmbed(ctx, inst.Addr(), sp.ID, promptTok); err != nil {
			res.Err = fmt.Errorf("warm-up: %w", err)
			return res
		}
		tps, err := benchEmbed(ctx, inst.Addr(), sp.ID, promptTok)
		if err != nil {
			res.Err = err
			return res
		}
		res.PromptTPS = tps
	}

	// Peak over a few snapshots; nvidia-smi lags slightly. Monitors that
	// measure only named processes (Apple Silicon) are told which one.
	if w, ok := mon.(gpu.PIDWatcher); ok {
		w.Watch([]int{inst.PID()})
	}
	for i := 0; i < 4; i++ {
		snap, err := mon.Snapshot(ctx)
		if err == nil {
			if mb := snap.UsedByPID(inst.PID()); mb > res.VRAMMB {
				res.VRAMMB = mb
			}
		}
		time.Sleep(250 * time.Millisecond)
	}
	if res.VRAMMB == 0 {
		res.Err = fmt.Errorf("gpu monitor reported no memory for pid %d", inst.PID())
	}
	return res
}

type throughput struct{ prompt, gen float64 }

func benchChat(ctx context.Context, addr, modelID string, promptTok, genTok int) (throughput, error) {
	// ~4 chars per token; a varied text so the prompt is not trivially cached.
	var sb strings.Builder
	for i := 0; sb.Len() < promptTok*4; i++ {
		fmt.Fprintf(&sb, "Paragraph %d: the scheduler weighs latency, memory and reuse before it loads or evicts a model. ", i)
	}
	body, _ := json.Marshal(map[string]any{
		"model":        modelID,
		"messages":     []map[string]string{{"role": "user", "content": sb.String() + "\n\nSummarize the above in detail."}},
		"max_tokens":   genTok,
		"temperature":  0,
		"cache_prompt": false,
	})
	raw, err := post(ctx, addr, "/v1/chat/completions", body)
	if err != nil {
		return throughput{}, err
	}
	var out struct {
		Timings struct {
			PromptPerSecond    float64 `json:"prompt_per_second"`
			PredictedPerSecond float64 `json:"predicted_per_second"`
		} `json:"timings"`
		Error json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return throughput{}, fmt.Errorf("bad response: %s", truncate(raw))
	}
	if len(out.Error) > 0 {
		return throughput{}, fmt.Errorf("upstream: %s", out.Error)
	}
	return throughput{out.Timings.PromptPerSecond, out.Timings.PredictedPerSecond}, nil
}

func benchEmbed(ctx context.Context, addr, modelID string, promptTok int) (float64, error) {
	docs := make([]string, 32)
	for i := range docs {
		docs[i] = fmt.Sprintf("Document %d discusses GPU memory residency and priority queues for local inference workloads.", i)
	}
	body, _ := json.Marshal(map[string]any{"model": modelID, "input": docs})
	start := time.Now()
	raw, err := post(ctx, addr, "/v1/embeddings", body)
	if err != nil {
		return 0, err
	}
	var out struct {
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
		Error json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return 0, fmt.Errorf("bad response: %s", truncate(raw))
	}
	if len(out.Error) > 0 {
		return 0, fmt.Errorf("upstream: %s", out.Error)
	}
	return float64(out.Usage.PromptTokens) / time.Since(start).Seconds(), nil
}

func post(ctx context.Context, addr, path string, body []byte) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+addr+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("upstream %d: %s", resp.StatusCode, truncate(raw))
	}
	return raw, nil
}

func truncate(b []byte) string {
	s := strings.TrimSpace(string(b))
	if len(s) > 300 {
		return s[:300] + "..."
	}
	return s
}

func pickFreePort(rng [2]int) (int, error) {
	for p := rng[0]; p <= rng[1]; p++ {
		ln, err := listenTCP(p)
		if err == nil {
			ln.Close()
			return p, nil
		}
	}
	return 0, fmt.Errorf("no free port in %d-%d", rng[0], rng[1])
}

func listenTCP(port int) (net.Listener, error) {
	return net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
}
