// Command chat is a minimal streaming chat client for demos. Tokens are
// printed as they arrive; a footer shows which model answered, how long the
// request waited for the GPU and the generation speed.
//
//	go run ./scripts/chat                          # interactive, default model
//	go run ./scripts/chat -model ornith -m "Write a haiku about VRAM."
package main

import (
	"bufio"
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
	esc   = "\033["
	reset = esc + "0m"
	dim   = esc + "2m"
	bold  = esc + "1m"
	cyan  = esc + "36m"
	red   = esc + "31m"
)

func main() {
	addr := flag.String("addr", envOr("GRIDCORE_ADDR", "127.0.0.1:8080"), "gridcore address")
	model := flag.String("model", "gpt-4o", "model id or alias")
	class := flag.String("class", "interactive", "workload class")
	maxTokens := flag.Int("max-tokens", 0, "cap the answer length; 0 (default) lets the model stop on its own")
	oneShot := flag.String("m", "", "send this message and exit")
	system := flag.String("system", "", "optional system prompt")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	var history []map[string]string
	if *system != "" {
		history = append(history, map[string]string{"role": "system", "content": *system})
	}
	ask := func(text string) {
		history = append(history, map[string]string{"role": "user", "content": text})
		answer, err := stream(ctx, *addr, *model, *class, *maxTokens, history)
		if err != nil {
			fmt.Printf("%s%v%s\n", red, err, reset)
			history = history[:len(history)-1]
			return
		}
		history = append(history, map[string]string{"role": "assistant", "content": answer})
	}

	if *oneShot != "" {
		fmt.Printf("%syou>%s %s\n", bold, reset, *oneShot)
		ask(*oneShot)
		return
	}
	fmt.Printf("%schat%s  model=%s  class=%s  (Ctrl-D to quit)\n\n", bold, reset, *model, *class)
	sc := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("%syou>%s ", bold, reset)
		if !sc.Scan() {
			fmt.Println()
			return
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		ask(line)
	}
}

// stream sends the request and prints tokens as they arrive; returns the
// full assistant text.
func stream(ctx context.Context, addr, model, class string, maxTokens int, messages []map[string]string) (string, error) {
	payload := map[string]any{"model": model, "messages": messages, "stream": true}
	if maxTokens > 0 {
		payload["max_tokens"] = maxTokens
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+addr+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-GridCore-Class", class)

	start := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, errorMessage(raw))
	}
	queued, _ := strconv.ParseInt(resp.Header.Get("X-GridCore-Queue-Ms"), 10, 64)
	served := resp.Header.Get("X-GridCore-Model")
	if served != "" {
		fmt.Printf("%s%s>%s ", cyan, served, reset)
	}
	var sb strings.Builder
	var firstToken time.Time
	var tokens int
	var genTPS float64
	var finish string
	rd := bufio.NewReader(resp.Body)
	for {
		line, err := rd.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimRight(line, "\r\n")
			// Long queue waits start the stream early; the wait then arrives
			// as an SSE comment instead of a header.
			if strings.HasPrefix(line, ": gridcore queue_ms=") {
				fields := strings.Fields(strings.TrimPrefix(line, ": gridcore "))
				for _, f := range fields {
					if v, ok := strings.CutPrefix(f, "queue_ms="); ok {
						queued, _ = strconv.ParseInt(v, 10, 64)
					}
					if v, ok := strings.CutPrefix(f, "model="); ok && served == "" {
						served = v
						fmt.Printf("%s%s>%s ", cyan, served, reset)
					}
				}
				continue
			}
			if strings.HasPrefix(line, "data: ") {
				payload := strings.TrimPrefix(line, "data: ")
				if payload == "[DONE]" {
					break
				}
				var ev struct {
					Choices []struct {
						FinishReason string `json:"finish_reason"`
						Delta        struct {
							Content          string `json:"content"`
							ReasoningContent string `json:"reasoning_content"`
						} `json:"delta"`
					} `json:"choices"`
					Timings struct {
						PredictedPerSecond float64 `json:"predicted_per_second"`
					} `json:"timings"`
					Error *struct {
						Message string `json:"message"`
					} `json:"error"`
				}
				if json.Unmarshal([]byte(payload), &ev) == nil {
					if ev.Error != nil {
						return sb.String(), fmt.Errorf("%s", ev.Error.Message)
					}
					if ev.Timings.PredictedPerSecond > 0 {
						genTPS = ev.Timings.PredictedPerSecond
					}
					for _, c := range ev.Choices {
						if c.FinishReason != "" {
							finish = c.FinishReason
						}
						if c.Delta.ReasoningContent != "" && tokens == 0 && firstToken.IsZero() {
							fmt.Print(dim + "…thinking" + reset + " ")
							firstToken = time.Now()
						}
						if c.Delta.Content != "" {
							if firstToken.IsZero() {
								firstToken = time.Now()
							}
							tokens++
							fmt.Print(c.Delta.Content)
							sb.WriteString(c.Delta.Content)
						}
					}
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return sb.String(), err
		}
	}
	total := time.Since(start)
	ttft := time.Duration(0)
	if !firstToken.IsZero() {
		ttft = firstToken.Sub(start)
	}
	footer := fmt.Sprintf("queued %s · first token %s · total %s", fmtDur(time.Duration(queued)*time.Millisecond), fmtDur(ttft), fmtDur(total))
	if genTPS > 0 {
		footer += fmt.Sprintf(" · %.0f tok/s", genTPS)
	}
	fmt.Printf("\n%s   [%s]%s\n", dim, footer, reset)
	if finish == "length" {
		if maxTokens > 0 {
			fmt.Printf("%s   ⚠ answer cut off at -max-tokens %d%s\n", "\033[33m", maxTokens, reset)
		} else {
			fmt.Printf("%s   ⚠ answer cut off: the model's context slot is full%s\n", "\033[33m", reset)
		}
	}
	fmt.Println()
	return sb.String(), nil
}

func errorMessage(raw []byte) string {
	var e struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &e) == nil && e.Error.Message != "" {
		return e.Error.Code + ": " + e.Error.Message
	}
	s := strings.TrimSpace(string(raw))
	if len(s) > 200 {
		s = s[:200] + "..."
	}
	return s
}

func fmtDur(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
