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
		Expected:   nil,
		SizeProbs:  []float64{1},
		OutputData: make(linalg.Vector, q.VectorSize),
	}
}

type queueState struct {
	Expected   []linalg.Vector
	SizeProbs  []float64
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

	res.Expected = make([]linalg.Vector, len(q.Expected)+1)
	res.SizeProbs = make([]float64, len(q.SizeProbs)+1)

	for i, old := range q.SizeProbs {
		res.SizeProbs[i] += old * flags[queueNop]
		if i > 0 {
			res.SizeProbs[i-1] += old * flags[queuePop]
		} else {
			res.SizeProbs[i] += old * flags[queuePop]
		}
		res.SizeProbs[i+1] += old * flags[queuePush]
	}

	for i, vec := range q.Expected {
		res.Expected[i] = vec.Copy().Scale(flags[queueNop] + flags[queuePush])
		if i > 0 {
			res.Expected[i-1].Add(vec.Copy().Scale(flags[queuePop]))
		}
	}

	pushData := ctrl[queueFlagCount:]
	for i, prob := range q.SizeProbs {
		pushVec := pushData.Copy().Scale(flags[queuePush] * prob)
		if i == len(res.Expected)-1 {
			res.Expected[i] = pushVec
		} else {
			res.Expected[i].Add(pushVec)
		}
	}

	res.OutputData = res.Expected[0]

	return &res
}
