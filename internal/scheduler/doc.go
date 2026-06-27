// Package scheduler owns all decisions: which job runs next, which model is
// loaded or evicted, and how VRAM is reserved.
//
// Design:
//
//   - A single goroutine owns all mutable state. API handlers enqueue a Job
//     and wait on a per-job channel; runtimes and the GPU monitor report via
//     events. No locks around policy code.
//   - Classes are strictly ordered (interactive > background > batch), FIFO
//     inside a class. No cross-class aging.
//   - GPU mode: while an interactive job runs (or ran within
//     interactive_idle_before_background), no new background/batch steps
//     start. Running steps finish.
//   - Planner for the head job: resident+free slot -> dispatch; resident,
//     slots busy -> wait; loading -> wait; fits free VRAM -> load; can evict
//     per residency rules -> drain+stop victims, load; else wait.
//   - VRAM reservations: a load in progress reserves its estimated size so
//     two decisions cannot over-commit the device.
//   - Invariant #1: an instance with running jobs is never stopped.
//
// Implemented in milestone M1 against the fake runtime and fake GPU.
package scheduler
