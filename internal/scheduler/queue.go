package scheduler

import "github.com/w512/gridcore/internal/job"

// queue holds waiting jobs, one FIFO per class. Classes are strictly
// ordered; there is no cross-class aging.
type queue struct {
	byClass map[job.Class][]*jobState
}

func newQueue() *queue {
	q := &queue{byClass: map[job.Class][]*jobState{}}
	for _, c := range job.Classes {
		q.byClass[c] = nil
	}
	return q
}

func (q *queue) push(js *jobState) {
	q.byClass[js.job.Class] = append(q.byClass[js.job.Class], js)
}

// remove deletes js from its class queue; returns false if absent.
func (q *queue) remove(js *jobState) bool {
	list := q.byClass[js.job.Class]
	for i, x := range list {
		if x == js {
			q.byClass[js.job.Class] = append(list[:i:i], list[i+1:]...)
			return true
		}
	}
	return false
}

// list returns the waiting jobs of class c in FIFO order (shared slice; do
// not mutate).
func (q *queue) list(c job.Class) []*jobState { return q.byClass[c] }

func (q *queue) len(c job.Class) int { return len(q.byClass[c]) }

func (q *queue) total() int {
	n := 0
	for _, l := range q.byClass {
		n += len(l)
	}
	return n
}

// all returns every waiting job, highest class first, FIFO within class.
func (q *queue) all() []*jobState {
	var out []*jobState
	for _, c := range job.Classes {
		out = append(out, q.byClass[c]...)
	}
	return out
}
