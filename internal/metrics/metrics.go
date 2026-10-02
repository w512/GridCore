// Package metrics defines every Prometheus series GridCore exports.
//
// Metrics are part of the product, not an afterthought: the scheduler's
// decisions are only trustworthy if they can be observed. Series names are
// fixed here so dashboards and the code cannot drift apart silently.
package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds all collectors. One instance per process.
type Metrics struct {
	reg *prometheus.Registry

	JobsTotal         *prometheus.CounterVec   // class, kind, outcome
	JobsRunning       *prometheus.GaugeVec     // class
	JobsQueued        *prometheus.GaugeVec     // class
	JobsCancelled     *prometheus.CounterVec   // class, phase
	QueueSeconds      *prometheus.HistogramVec // class
	ExecutionSeconds  *prometheus.HistogramVec // class, model
	ModelLoadSeconds  *prometheus.HistogramVec // model
	ModelLoadsTotal   *prometheus.CounterVec   // model, outcome
	ModelEvictions    *prometheus.CounterVec   // model, reason
	ModelThrash       *prometheus.CounterVec   // model
	VariantSelected   *prometheus.CounterVec   // family, variant, class
	ModelsResident    prometheus.Gauge
	InstanceCrashes   *prometheus.CounterVec // model
	GPUVRAMUsedBytes  prometheus.Gauge
	GPUVRAMReserved   prometheus.Gauge
	GPUVRAMTotalBytes prometheus.Gauge
	GPUUtilization    prometheus.Gauge
	MemoryPressure    prometheus.Gauge       // unified memory: 0 unknown, 1 normal, 2 warn, 4 critical
	DeviceOOMs        *prometheus.CounterVec // model
	TokensTotal       *prometheus.CounterVec // model, direction (prompt|generated)
}

// New creates and registers all collectors in a private registry.
func New() *Metrics {
	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))

	f := promauto{reg}
	queueBuckets := []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120}
	execBuckets := []float64{.05, .1, .25, .5, 1, 2.5, 5, 10, 20, 40, 80, 160}
	loadBuckets := []float64{.5, 1, 2, 4, 8, 15, 30, 60, 120}

	return &Metrics{
		reg:               reg,
		JobsTotal:         f.counterVec("gridcore_jobs_total", "Jobs finished, by class, kind and outcome.", "class", "kind", "outcome"),
		JobsRunning:       f.gaugeVec("gridcore_jobs_running", "Jobs currently executing.", "class"),
		JobsQueued:        f.gaugeVec("gridcore_jobs_queued", "Jobs waiting for a slot or a model.", "class"),
		JobsCancelled:     f.counterVec("gridcore_jobs_cancelled_total", "Jobs cancelled by the client, by phase (queued|running).", "class", "phase"),
		QueueSeconds:      f.histVec("gridcore_queue_seconds", "Time from enqueue to first dispatch.", queueBuckets, "class"),
		ExecutionSeconds:  f.histVec("gridcore_execution_seconds", "Time from dispatch to completion.", execBuckets, "class", "model"),
		ModelLoadSeconds:  f.histVec("gridcore_model_load_seconds", "Time from spawn to healthy.", loadBuckets, "model"),
		ModelLoadsTotal:   f.counterVec("gridcore_model_loads_total", "Model load attempts by outcome (ok|error|timeout).", "model", "outcome"),
		ModelEvictions:    f.counterVec("gridcore_model_evictions_total", "Models unloaded to make room, by reason.", "model", "reason"),
		ModelThrash:       f.counterVec("gridcore_model_thrash_total", "Times a model was found reloading too often (the models in use do not fit together).", "model"),
		VariantSelected:   f.counterVec("gridcore_variant_selected_total", "Family requests by the variant that served them.", "family", "variant", "class"),
		ModelsResident:    f.gauge("gridcore_models_resident", "Models currently loaded."),
		InstanceCrashes:   f.counterVec("gridcore_instance_crashes_total", "Runtime processes that exited unexpectedly.", "model"),
		GPUVRAMUsedBytes:  f.gauge("gridcore_gpu_vram_used_bytes", "VRAM in use on the managed device (all processes)."),
		GPUVRAMReserved:   f.gauge("gridcore_gpu_vram_reserved_bytes", "VRAM reserved by the scheduler for loads in progress."),
		GPUVRAMTotalBytes: f.gauge("gridcore_gpu_vram_total_bytes", "VRAM budget (device total or configured limit)."),
		GPUUtilization:    f.gauge("gridcore_gpu_utilization_percent", "GPU compute utilisation, 0-100."),
		MemoryPressure:    f.gauge("gridcore_memory_pressure_level", "Host memory pressure acted on, unified memory only: 0 unknown, 1 normal, 2 warn, 4 critical."),
		DeviceOOMs:        f.counterVec("gridcore_device_oom_total", "Out-of-memory errors reported by running instances (an overcommitted unified-memory device).", "model"),
		TokensTotal:       f.counterVec("gridcore_tokens_total", "Tokens processed, by model and direction.", "model", "direction"),
	}
}

// Handler serves the registry in Prometheus text format.
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.reg, promhttp.HandlerOpts{})
}

// Registry exposes the underlying registry (tests, extra collectors).
func (m *Metrics) Registry() *prometheus.Registry { return m.reg }

type promauto struct{ reg *prometheus.Registry }

func (p promauto) counterVec(name, help string, labels ...string) *prometheus.CounterVec {
	c := prometheus.NewCounterVec(prometheus.CounterOpts{Name: name, Help: help}, labels)
	p.reg.MustRegister(c)
	return c
}

func (p promauto) gaugeVec(name, help string, labels ...string) *prometheus.GaugeVec {
	g := prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: name, Help: help}, labels)
	p.reg.MustRegister(g)
	return g
}

func (p promauto) gauge(name, help string) prometheus.Gauge {
	g := prometheus.NewGauge(prometheus.GaugeOpts{Name: name, Help: help})
	p.reg.MustRegister(g)
	return g
}

func (p promauto) histVec(name, help string, buckets []float64, labels ...string) *prometheus.HistogramVec {
	h := prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: name, Help: help, Buckets: buckets}, labels)
	p.reg.MustRegister(h)
	return h
}
