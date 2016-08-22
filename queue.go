package neuralstruct

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const queueFlagCount = 3

const (
	queueNop int = iota
	queuePush
	queuePop
)

// A Queue is a differentiable probabilistic FIFO queue
// of real-valued vectors.
type Queue struct {
	VectorSize int
}

// ControlSize returns the number of control vector
// components, which varies with q.VectorSize.
func (q *Queue) ControlSize() int {
	return q.VectorSize + queueFlagCount
}

// DataSize returns the size of vectors stored in
// this queue, as determined by q.VectorSize.
func (q *Queue) DataSize() int {
	return q.VectorSize
}

// StartState returns a state representing an empty queue.
func (q *Queue) StartState() State {
	return &queueState{
		Queues:     [][]linalg.Vector{nil},
		Probs:      []float64{1},
		OutputData: make(linalg.Vector, q.VectorSize),
	}
}

type queueState struct {
	Queues     [][]linalg.Vector
	Probs      []float64
	OutputData linalg.Vector
}

func (q *queueState) Data() linalg.Vector {
	return q.OutputData
}

func (q *queueState) Gradient(dataGrad linalg.Vector, upstreamGrad Grad) (linalg.Vector, Grad) {
	// TODO: this.
	return nil, nil
}

func (q *queueState) NextState(ctrl linalg.Vector) State {
	probs := ctrl[:queueFlagCount]
	softmax := autofunc.Softmax{}
	flags := softmax.Apply(&autofunc.Variable{Vector: probs}).Output()

	var res queueState

	res.Queues = append(res.Queues, q.Queues...)
	for _, prob := range q.Probs {
		res.Probs = append(res.Probs, prob*flags[queueNop])
	}

	for i, queue := range q.Queues {
		newQueue := make([]linalg.Vector, len(queue)+1)
		copy(newQueue, queue)
		newQueue[len(newQueue)-1] = ctrl[queueFlagCount:]
		res.Queues = append(res.Queues, newQueue)
		res.Probs = append(res.Probs, q.Probs[i]*flags[queuePush])
	}

	for i, queue := range q.Queues {
		if len(queue) == 0 {
			res.Queues = append(res.Queues, queue)
		} else {
			res.Queues = append(res.Queues, queue[1:])
		}
		res.Probs = append(res.Probs, q.Probs[i]*flags[queuePop])
	}

	expectedData := make(linalg.Vector, len(q.Data()))
	for i, queue := range res.Queues {
		if len(queue) == 0 {
			continue
		}
		expectedData.Add(queue[0].Copy().Scale(res.Probs[i]))
	}
	res.OutputData = expectedData

	return &res
}
