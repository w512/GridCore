package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/w512/gridcore/internal/job"
	"github.com/w512/gridcore/internal/scheduler"
)

// keepaliveAfter is how long a streaming request may sit in the queue before
// we commit a 200/text-event-stream and start sending SSE comments. Until
// then the real upstream status can still be passed through.
const keepaliveAfter = 5 * time.Second

// keepaliveEvery is the interval between SSE comments once started.
const keepaliveEvery = 5 * time.Second

// proxy holds the upstream HTTP client.
type proxy struct {
	client *http.Client
}

func newProxy() *proxy {
	return &proxy{client: &http.Client{
		// No overall timeout: generations can run for minutes and streaming
		// responses stay open. Cancellation comes from the request context.
		Transport: &http.Transport{
			DialContext: (&net.Dialer{Timeout: 5 * time.Second}).DialContext,
			// llama-server (cpp-httplib) closes a keep-alive connection after
			// a handful of requests and a few idle seconds; reusing one that
			// the server already closed cost us a failed request each time.
			// Connections are loopback, so a fresh one per request is cheap.
			DisableKeepAlives: true,
			// llama-server sends chunked SSE; never buffer.
			DisableCompression: true,
		},
	}}
}

// doWithRetry sends a request whose body is replayable and retries once
// when the transport fails before any response arrived (typically a
// keep-alive connection the server closed underneath us).
func (p *proxy) doWithRetry(ctx context.Context, method, url string, body []byte, hdr http.Header) (*http.Response, error) {
	var lastErr error
	for attempt := 0; attempt < 2; attempt++ {
		req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		for k, v := range hdr {
			req.Header[k] = v
		}
		resp, err := p.client.Do(req)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if ctx.Err() != nil || !isRetryableTransport(err) {
			break
		}
		slog.Warn("upstream transport error, retrying once", "url", url, "err", err)
	}
	return nil, lastErr
}

// logUpstreamErr logs a failed upstream call; a client that went away is
// routine and only logged at debug level.
func logUpstreamErr(ctx context.Context, addr, path string, err error) {
	if ctx.Err() != nil {
		slog.Debug("upstream request cancelled by client", "addr", addr, "path", path)
		return
	}
	slog.Warn("upstream unreachable", "addr", addr, "path", path, "err", err)
}

func isRetryableTransport(err error) bool {
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.EPIPE) {
		return true
	}
	s := err.Error()
	return strings.Contains(s, "connection reset") || strings.Contains(s, "EOF") || strings.Contains(s, "broken pipe") || strings.Contains(s, "server closed idle connection")
}

// inference is the shared handler for the three OpenAI endpoints.
func (s *Server) inference(kind job.Kind) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.sched == nil {
			WriteError(w, http.StatusServiceUnavailable, "server_error", "no_scheduler", "scheduler not running")
			return
		}
		req, aerr := parseInferenceRequest(r, kind, s.cfg, s.MaxBodyBytes)
		if aerr != nil {
			WriteError(w, aerr.status, aerr.typ, aerr.code, aerr.msg)
			return
		}

		j := job.New(newJobID(), r.Context(), req.class, kind, req.modelID)
		j.MaxWait = req.maxWait
		j.Steps = req.steps
		h, err := s.sched.Submit(j)
		if err != nil {
			if errors.Is(err, scheduler.ErrShuttingDown) {
				WriteError(w, http.StatusServiceUnavailable, "server_error", "shutting_down", "GridCore is shutting down")
				return
			}
			WriteError(w, http.StatusBadRequest, "invalid_request_error", "submit_failed", err.Error())
			return
		}
		w.Header().Set(HeaderJobID, j.ID)
		w.Header().Set(HeaderModel, req.modelID)

		if req.steps == 1 {
			s.single(w, r, req, h)
		} else {
			s.chunked(w, r, req, h)
		}
	}
}

// single proxies a one-step job: wait for the grant, forward the request,
// stream the response back.
func (s *Server) single(w http.ResponseWriter, r *http.Request, req *inferenceRequest, h *scheduler.Handle) {
	g, committed, err := waitGrant(w, r, req, h)
	if err != nil {
		abandon(h, err)
		writeJobError(w, committed, err)
		return
	}
	if !committed {
		w.Header().Set(HeaderQueueMS, strconv.FormatInt(g.Queued.Milliseconds(), 10))
	} else {
		// Headers are gone (SSE keep-alive started); report the wait as a
		// comment so streaming clients can still show it.
		_, _ = fmt.Fprintf(w, ": gridcore queue_ms=%d model=%s\n\n", g.Queued.Milliseconds(), g.Model)
		_ = http.NewResponseController(w).Flush()
	}
	res := s.proxy.forward(r.Context(), w, committed, g.Addr, r.URL.Path, req.marshalBody(), req.stream)
	h.StepDone(g.Step, res.err, res.usage)
}

// chunked runs a multi-step embeddings job: every grant fans out one chunk,
// results are reassembled in input order.
func (s *Server) chunked(w http.ResponseWriter, r *http.Request, req *inferenceRequest, h *scheduler.Handle) {
	size := s.cfg.Policy.EmbeddingChunkSize
	type chunkResult struct {
		data  []embeddingItem
		usage embeddingUsage
		model string
	}
	results := make([]*chunkResult, req.steps)
	var (
		mu       sync.Mutex
		wg       sync.WaitGroup
		firstErr error
		queued   time.Duration
		gotFirst bool
	)
	ctx := r.Context()

	runChunk := func(g scheduler.Grant) {
		defer wg.Done()
		lo := g.Step * size
		hi := min(lo+size, len(req.input))
		body := req.chunkBody(lo, hi)
		var out struct {
			Data  []embeddingItem `json:"data"`
			Usage embeddingUsage  `json:"usage"`
			Model string          `json:"model"`
		}
		status, raw, err := s.proxy.call(ctx, g.Addr, r.URL.Path, body)
		var u scheduler.Usage
		switch {
		case err != nil:
		case status/100 != 2:
			err = &upstreamError{status: status, body: raw}
		case json.Unmarshal(raw, &out) != nil:
			err = fmt.Errorf("upstream returned invalid embeddings JSON")
		default:
			for i := range out.Data {
				out.Data[i].Index = lo + i
			}
			u = scheduler.Usage{PromptTokens: out.Usage.PromptTokens}
			mu.Lock()
			results[g.Step] = &chunkResult{data: out.Data, usage: out.Usage, model: out.Model}
			mu.Unlock()
		}
		if err != nil {
			mu.Lock()
			if firstErr == nil {
				firstErr = err
			}
			mu.Unlock()
		}
		h.StepDone(g.Step, err, u)
	}

	for {
		select {
		case g := <-h.Grants():
			if !gotFirst {
				gotFirst = true
				queued = g.Queued
			}
			wg.Add(1)
			go runChunk(g)
		case <-h.Done():
			wg.Wait()
			abandon(h, h.Err()) // grants delivered before the failure must not leak slots
			if err := h.Err(); err != nil {
				mu.Lock()
				up := firstErr
				mu.Unlock()
				if up != nil {
					writeUpstreamError(w, up)
					return
				}
				writeJobError(w, false, err)
				return
			}
			// All steps completed: assemble.
			var data []embeddingItem
			var usage embeddingUsage
			model := req.modelID
			for _, cr := range results {
				if cr == nil {
					writeJobError(w, false, errors.New("internal: missing chunk result"))
					return
				}
				data = append(data, cr.data...)
				usage.PromptTokens += cr.usage.PromptTokens
				usage.TotalTokens += cr.usage.TotalTokens
				if cr.model != "" {
					model = cr.model
				}
			}
			sort.Slice(data, func(i, j int) bool { return data[i].Index < data[j].Index })
			w.Header().Set(HeaderQueueMS, strconv.FormatInt(queued.Milliseconds(), 10))
			writeJSON(w, http.StatusOK, map[string]any{
				"object": "list", "data": data, "model": model, "usage": usage,
			})
			return
		}
	}
}

type embeddingItem struct {
	Object    string          `json:"object"`
	Index     int             `json:"index"`
	Embedding json.RawMessage `json:"embedding"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// abandon returns any grants already sitting in the channel to the
// scheduler as failed steps. Without this a client that disconnects at the
// same instant a grant is issued would leave the slot busy forever.
func abandon(h *scheduler.Handle, cause error) {
	if cause == nil {
		cause = errors.New("abandoned")
	}
	for {
		select {
		case g := <-h.Grants():
			h.StepDone(g.Step, cause, scheduler.Usage{})
		default:
			return
		}
	}
}

// waitGrant blocks until the job gets its first grant or terminates. For
// streaming requests that wait longer than keepaliveAfter it commits an SSE
// response and emits comments so clients and proxies keep the connection.
// committed reports whether headers were already written.
func waitGrant(w http.ResponseWriter, r *http.Request, req *inferenceRequest, h *scheduler.Handle) (g scheduler.Grant, committed bool, err error) {
	first := time.NewTimer(keepaliveAfter)
	defer first.Stop()
	var tick *time.Ticker
	defer func() {
		if tick != nil {
			tick.Stop()
		}
	}()
	var tickC <-chan time.Time
	rc := http.NewResponseController(w)

	for {
		select {
		case g = <-h.Grants():
			return g, committed, nil
		case <-h.Done():
			if err := h.Err(); err != nil {
				return scheduler.Grant{}, committed, err
			}
			return scheduler.Grant{}, committed, errors.New("job finished without a grant")
		case <-first.C:
			if !req.stream {
				continue
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("X-Accel-Buffering", "no")
			w.WriteHeader(http.StatusOK)
			committed = true
			_, _ = io.WriteString(w, ": gridcore queued\n\n")
			_ = rc.Flush()
			tick = time.NewTicker(keepaliveEvery)
			tickC = tick.C
		case <-tickC:
			_, _ = io.WriteString(w, ": gridcore queued\n\n")
			_ = rc.Flush()
		case <-r.Context().Done():
			// Client went away. Tell the scheduler and wait until it has
			// stopped dispatching; grants that were already on their way are
			// handed back so no slot leaks.
			h.Cancel()
			settle := time.NewTimer(10 * time.Second)
			defer settle.Stop()
			for {
				select {
				case g := <-h.Grants():
					h.StepDone(g.Step, scheduler.ErrCancelled, scheduler.Usage{})
				case <-h.Done():
					return scheduler.Grant{}, committed, scheduler.ErrCancelled
				case <-settle.C:
					return scheduler.Grant{}, committed, scheduler.ErrCancelled
				}
			}
		}
	}
}

// stepResult is the outcome of one forwarded request.
type stepResult struct {
	status int
	usage  scheduler.Usage
	err    error // transport failure or non-2xx upstream
}

// forward sends body to addr+path and relays the response to w. When
// committed is true the SSE response has already started, so upstream errors
// are delivered as SSE error events.
func (p *proxy) forward(ctx context.Context, w http.ResponseWriter, committed bool, addr, path string, body []byte, stream bool) stepResult {
	hdr := http.Header{"Content-Type": {"application/json"}}
	if stream {
		hdr.Set("Accept", "text/event-stream")
	}
	resp, err := p.doWithRetry(ctx, http.MethodPost, "http://"+addr+path, body, hdr)
	if err != nil {
		logUpstreamErr(ctx, addr, path, err)
		if !committed {
			if ctx.Err() == nil {
				WriteError(w, http.StatusBadGateway, "server_error", "upstream_unreachable", "inference runtime did not respond: "+err.Error())
			}
		} else {
			writeSSEError(w, "upstream_unreachable", err.Error())
		}
		return stepResult{err: err}
	}
	defer resp.Body.Close()
	res := stepResult{status: resp.StatusCode}
	rc := http.NewResponseController(w)

	if committed {
		// Headers are out already. Relay SSE bytes, or wrap an error.
		if resp.StatusCode/100 != 2 {
			raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
			writeSSEError(w, "upstream_error", fmt.Sprintf("upstream %d: %s", resp.StatusCode, bytes.TrimSpace(raw)))
			res.err = &upstreamError{status: resp.StatusCode, body: raw}
			return res
		}
		res.usage, res.err = relay(w, rc, resp.Body)
		return res
	}

	// Pass upstream status and content headers through.
	for _, hname := range []string{"Content-Type", "Cache-Control", "X-Accel-Buffering"} {
		if v := resp.Header.Get(hname); v != "" {
			w.Header().Set(hname, v)
		}
	}
	if resp.StatusCode/100 != 2 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		w.Header().Set("Content-Length", strconv.Itoa(len(raw)))
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(raw)
		res.err = &upstreamError{status: resp.StatusCode, body: raw}
		return res
	}
	if stream {
		w.WriteHeader(resp.StatusCode)
		res.usage, res.err = relay(w, rc, resp.Body)
		return res
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		res.err = err
		if ctx.Err() == nil {
			WriteError(w, http.StatusBadGateway, "server_error", "upstream_read_failed", err.Error())
		}
		return res
	}
	res.usage = extractUsage(raw)
	w.Header().Set("Content-Length", strconv.Itoa(len(raw)))
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(raw)
	return res
}

// relay copies an SSE body flushing after every read so tokens reach the
// client as soon as the runtime produces them. It also watches the stream
// for the final chunk: llama-server attaches "timings" (and "usage" when
// stream_options.include_usage is set) to it, which feeds the profile.
func relay(w http.ResponseWriter, rc *http.ResponseController, body io.Reader) (scheduler.Usage, error) {
	buf := make([]byte, 32<<10)
	var last sseTail
	for {
		n, err := body.Read(buf)
		if n > 0 {
			last.observe(buf[:n])
			if _, werr := w.Write(buf[:n]); werr != nil {
				return last.usage(), werr
			}
			_ = rc.Flush()
		}
		if err == io.EOF {
			return last.usage(), nil
		}
		if err != nil {
			return last.usage(), err
		}
	}
}

// sseTail keeps the most recent complete "data:" payload that carries usage
// or timings. It only looks at line boundaries, so it is cheap.
type sseTail struct {
	pending []byte
	found   []byte
}

func (t *sseTail) observe(p []byte) {
	t.pending = append(t.pending, p...)
	for {
		i := bytes.IndexByte(t.pending, '\n')
		if i < 0 {
			break
		}
		line := bytes.TrimRight(t.pending[:i], "\r")
		t.pending = t.pending[i+1:]
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		payload := bytes.TrimSpace(line[5:])
		if bytes.Contains(payload, []byte(`"timings"`)) || bytes.Contains(payload, []byte(`"usage"`)) {
			t.found = append(t.found[:0], payload...)
		}
	}
	if len(t.pending) > 1<<20 {
		t.pending = t.pending[len(t.pending)-1<<20:] // never grow without bound
	}
}

func (t *sseTail) usage() scheduler.Usage {
	if len(t.found) == 0 {
		return scheduler.Usage{}
	}
	return extractUsage(t.found)
}

// call performs a request and returns status and body (chunked embeddings).
func (p *proxy) call(ctx context.Context, addr, path string, body []byte) (int, []byte, error) {
	resp, err := p.doWithRetry(ctx, http.MethodPost, "http://"+addr+path, body, http.Header{"Content-Type": {"application/json"}})
	if err != nil {
		logUpstreamErr(ctx, addr, path, err)
		return 0, nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	return resp.StatusCode, raw, err
}

// extractUsage pulls token counts and llama.cpp timings out of a response.
func extractUsage(raw []byte) scheduler.Usage {
	var v struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
		Timings struct {
			PromptPerSecond    float64 `json:"prompt_per_second"`
			PredictedPerSecond float64 `json:"predicted_per_second"`
		} `json:"timings"`
	}
	if json.Unmarshal(raw, &v) != nil {
		return scheduler.Usage{}
	}
	return scheduler.Usage{
		PromptTokens:     v.Usage.PromptTokens,
		CompletionTokens: v.Usage.CompletionTokens,
		PromptTPS:        v.Timings.PromptPerSecond,
		GenTPS:           v.Timings.PredictedPerSecond,
	}
}

// upstreamError carries a non-2xx runtime response.
type upstreamError struct {
	status int
	body   []byte
}

func (e *upstreamError) Error() string { return fmt.Sprintf("upstream status %d", e.status) }

// StatusCode lets the scheduler tell client errors (4xx) from runtime
// failures (5xx) when labelling outcomes.
func (e *upstreamError) StatusCode() int { return e.status }

func writeUpstreamError(w http.ResponseWriter, err error) {
	var up *upstreamError
	if errors.As(err, &up) {
		ct := "application/json"
		if !json.Valid(up.body) {
			ct = "text/plain"
		}
		w.Header().Set("Content-Type", ct)
		w.WriteHeader(up.status)
		_, _ = w.Write(up.body)
		return
	}
	WriteError(w, http.StatusBadGateway, "server_error", "upstream_error", err.Error())
}

// writeJobError maps scheduler outcomes to HTTP.
func writeJobError(w http.ResponseWriter, committed bool, err error) {
	status, code, msg := http.StatusInternalServerError, "internal_error", err.Error()
	typ := "server_error"
	switch {
	case errors.Is(err, scheduler.ErrCancelled):
		return // client is gone
	case errors.Is(err, scheduler.ErrQueueTimeout):
		status, code = http.StatusServiceUnavailable, "queue_timeout"
		msg = "request waited longer than max_wait; retry later or lower the load"
		w.Header().Set("Retry-After", "5")
	case errors.Is(err, scheduler.ErrModelDisabled):
		status, code = http.StatusServiceUnavailable, "model_disabled"
	case errors.Is(err, scheduler.ErrModelTooLarge):
		status, code = http.StatusServiceUnavailable, "model_too_large"
	case errors.Is(err, scheduler.ErrLoadFailed):
		status, code = http.StatusBadGateway, "model_load_failed"
	case errors.Is(err, scheduler.ErrInstanceFailed):
		status, code = http.StatusBadGateway, "instance_failed"
	case errors.Is(err, scheduler.ErrShuttingDown):
		status, code = http.StatusServiceUnavailable, "shutting_down"
	case errors.Is(err, scheduler.ErrStepFailed):
		status, code = http.StatusBadGateway, "upstream_error"
	}
	if committed {
		writeSSEError(w, code, msg)
		return
	}
	WriteError(w, status, typ, code, msg)
}

// writeSSEError emits an OpenAI-style error event on an open SSE stream.
func writeSSEError(w http.ResponseWriter, code, msg string) {
	payload, _ := json.Marshal(errorBody("server_error", code, msg))
	_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", payload)
	_ = http.NewResponseController(w).Flush()
}
