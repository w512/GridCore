package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gridcore/gridcore/internal/config"
	"github.com/gridcore/gridcore/internal/job"
)

// Request headers understood by GridCore.
const (
	HeaderClass   = "X-GridCore-Class"
	HeaderMaxWait = "X-GridCore-Max-Wait-Ms"
	HeaderJobID   = "X-GridCore-Job-Id"
	HeaderModel   = "X-GridCore-Model"
	HeaderQueueMS = "X-GridCore-Queue-Ms"
)

// apiError is a client-facing error with an HTTP status.
type apiError struct {
	status int
	typ    string
	code   string
	msg    string
}

func (e *apiError) Error() string { return e.msg }

func badRequest(code, msg string) *apiError {
	return &apiError{status: http.StatusBadRequest, typ: "invalid_request_error", code: code, msg: msg}
}

// inferenceRequest is a parsed and validated inference call.
type inferenceRequest struct {
	kind    job.Kind
	class   job.Class
	maxWait time.Duration
	modelID string
	stream  bool

	// body is the request with GridCore fields stripped and the model
	// rewritten to the resolved id. Re-marshalled per step.
	body map[string]json.RawMessage
	// input holds the embeddings input array when it is an array.
	input []json.RawMessage
	steps int
}

// parseInferenceRequest reads and validates the body. Class precedence:
// header, body "gridcore" object, "model@class" suffix, policy default.
func parseInferenceRequest(r *http.Request, kind job.Kind, cfg *config.Config, maxBody int64) (*inferenceRequest, *apiError) {
	raw, err := io.ReadAll(http.MaxBytesReader(nil, r.Body, maxBody))
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return nil, &apiError{status: http.StatusRequestEntityTooLarge, typ: "invalid_request_error", code: "body_too_large",
				msg: fmt.Sprintf("request body exceeds %d bytes", maxBody)}
		}
		return nil, badRequest("invalid_body", "could not read request body: "+err.Error())
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, badRequest("invalid_json", "request body is not a JSON object: "+err.Error())
	}

	req := &inferenceRequest{kind: kind, body: body, steps: 1}

	// model (required) and optional @class suffix
	var modelName string
	if rawModel, ok := body["model"]; ok {
		if err := json.Unmarshal(rawModel, &modelName); err != nil {
			return nil, badRequest("invalid_model", "model must be a string")
		}
	}
	if modelName == "" {
		return nil, badRequest("model_required", "model is required")
	}
	var suffixClass string
	if name, cls, found := strings.Cut(modelName, "@"); found {
		modelName, suffixClass = name, cls
	}

	// body "gridcore": {"class": "...", "max_wait_ms": N}
	var bodyOpts struct {
		Class     string `json:"class"`
		MaxWaitMS *int64 `json:"max_wait_ms"`
	}
	if rawOpts, ok := body["gridcore"]; ok {
		if err := json.Unmarshal(rawOpts, &bodyOpts); err != nil {
			return nil, badRequest("invalid_gridcore", `"gridcore" must be an object like {"class":"background","max_wait_ms":5000}`)
		}
		delete(body, "gridcore")
	}

	// class precedence
	classStr := string(cfg.Policy.DefaultClass)
	switch {
	case r.Header.Get(HeaderClass) != "":
		classStr = r.Header.Get(HeaderClass)
	case bodyOpts.Class != "":
		classStr = bodyOpts.Class
	case suffixClass != "":
		classStr = suffixClass
	}
	class, err := job.ParseClass(strings.ToLower(strings.TrimSpace(classStr)))
	if err != nil {
		return nil, badRequest("invalid_class", err.Error())
	}
	req.class = class

	// max_wait precedence: header, body
	if v := r.Header.Get(HeaderMaxWait); v != "" {
		ms, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		if err != nil || ms < 0 {
			return nil, badRequest("invalid_max_wait", HeaderMaxWait+" must be a non-negative integer")
		}
		req.maxWait = time.Duration(ms) * time.Millisecond
	} else if bodyOpts.MaxWaitMS != nil {
		if *bodyOpts.MaxWaitMS < 0 {
			return nil, badRequest("invalid_max_wait", "gridcore.max_wait_ms must be non-negative")
		}
		req.maxWait = time.Duration(*bodyOpts.MaxWaitMS) * time.Millisecond
	}

	// model resolution and capability
	id, ok := cfg.ResolveModel(modelName)
	if !ok {
		return nil, &apiError{status: http.StatusNotFound, typ: "invalid_request_error", code: "model_not_found",
			msg: fmt.Sprintf("model %q is not configured", modelName)}
	}
	req.modelID = id
	m := cfg.Models[id]
	if !capabilityOK(kind, m) {
		return nil, badRequest("model_capability",
			fmt.Sprintf("model %q (capabilities %v) cannot serve %s", id, m.Capabilities, endpointName(kind)))
	}
	body["model"], _ = json.Marshal(id)

	if rawStream, ok := body["stream"]; ok {
		_ = json.Unmarshal(rawStream, &req.stream)
	}

	// embeddings: chunk large inputs
	if kind == job.Embedding {
		rawInput, ok := body["input"]
		if !ok {
			return nil, badRequest("input_required", "input is required")
		}
		var arr []json.RawMessage
		if err := json.Unmarshal(rawInput, &arr); err == nil {
			if len(arr) == 0 {
				return nil, badRequest("input_empty", "input must not be empty")
			}
			req.input = arr
			if n, size := len(arr), cfg.Policy.EmbeddingChunkSize; n > size {
				req.steps = (n + size - 1) / size
			}
		}
	}
	return req, nil
}

// capabilityOK implements the endpoint -> capability table:
//
//	/v1/chat/completions  chat
//	/v1/completions       completion or chat
//	/v1/embeddings        embedding
func capabilityOK(kind job.Kind, m config.Model) bool {
	switch kind {
	case job.Chat:
		return m.HasCapability(config.CapChat)
	case job.Completion:
		return m.HasCapability(config.CapCompletion) || m.HasCapability(config.CapChat)
	case job.Embedding:
		return m.HasCapability(config.CapEmbedding)
	}
	return false
}

func endpointName(kind job.Kind) string {
	switch kind {
	case job.Chat:
		return "/v1/chat/completions"
	case job.Completion:
		return "/v1/completions"
	case job.Embedding:
		return "/v1/embeddings"
	}
	return string(kind)
}

// marshalBody re-encodes the (possibly modified) body.
func (req *inferenceRequest) marshalBody() []byte {
	b, _ := json.Marshal(req.body)
	return b
}

// chunkBody returns the body with input replaced by items [lo, hi).
func (req *inferenceRequest) chunkBody(lo, hi int) []byte {
	clone := make(map[string]json.RawMessage, len(req.body))
	for k, v := range req.body {
		clone[k] = v
	}
	clone["input"], _ = json.Marshal(req.input[lo:hi])
	b, _ := json.Marshal(clone)
	return b
}
