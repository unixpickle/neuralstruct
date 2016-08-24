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

	ControlIn linalg.Vector
	Last      *queueState
}

func (q *queueState) Data() linalg.Vector {
	return q.OutputData
}

func (q *queueState) Gradient(dataGrad linalg.Vector, upstreamGrad Grad) (linalg.Vector, Grad) {
	if q.Last == nil {
		panic("cannot propagate through start state")
	}
	softmax := autofunc.Softmax{}
	flagsVar := &autofunc.Variable{Vector: q.ControlIn[:queueFlagCount]}
	flagRes := softmax.Apply(flagsVar)
	flags := flagRes.Output()

	flagsGrad := make(linalg.Vector, queueFlagCount)

	var upstream *queueUpstream
	if upstreamGrad != nil {
		upstream = upstreamGrad.(*queueUpstream)
		upstream.Expected[0].Add(dataGrad)
	} else {
		upstream = new(queueUpstream)
		upstream.SizeProbs = make([]float64, len(q.SizeProbs))
		upstream.Expected = make([]linalg.Vector, len(q.Expected))
		upstream.Expected[0] = dataGrad
		zeroVec := make(linalg.Vector, len(dataGrad))
		for i := 1; i < len(upstream.Expected); i++ {
			upstream.Expected[i] = zeroVec
		}
	}

	downstream := &queueUpstream{
		Expected:  make([]linalg.Vector, len(q.Last.Expected)),
		SizeProbs: make([]float64, len(q.Last.SizeProbs)),
	}
	for i, vec := range q.Last.Expected {
		downstream.Expected[i] = upstream.Expected[i].Copy().Scale(flags[queueNop] +
			flags[queuePush])
		flagsGrad[queueNop] += vec.Dot(upstream.Expected[i])
		flagsGrad[queuePush] += vec.Dot(upstream.Expected[i])
		if i > 0 {
			downstream.Expected[i].Add(upstream.Expected[i-1].Copy().Scale(flags[queuePop]))
			flagsGrad[queuePop] += vec.Dot(upstream.Expected[i-1])
		}
	}

	pushDataGrad := make(linalg.Vector, len(q.Data()))

	for i, prob := range q.Last.SizeProbs {
		pushData := q.ControlIn[queueFlagCount:]
		pushDataGrad.Add(upstream.Expected[i].Copy().Scale(flags[queuePush] * prob))
		upstreamDot := upstream.Expected[i].Dot(pushData)
		flagsGrad[queuePush] += prob * upstreamDot
		downstream.SizeProbs[i] += flags[queuePush] * upstreamDot
	}

	for i, old := range q.Last.SizeProbs {
		downstream.SizeProbs[i] += flags[queueNop] * upstream.SizeProbs[i]
		flagsGrad[queueNop] += old * upstream.SizeProbs[i]
		if i > 0 {
			downstream.SizeProbs[i] += flags[queuePop] * upstream.SizeProbs[i-1]
			flagsGrad[queuePop] += old * upstream.SizeProbs[i-1]
		} else {
			downstream.SizeProbs[i] += flags[queuePop] * upstream.SizeProbs[i]
			flagsGrad[queuePop] += old * upstream.SizeProbs[i]
		}
		downstream.SizeProbs[i] += flags[queuePush] * upstream.SizeProbs[i+1]
		flagsGrad[queuePush] += old * upstream.SizeProbs[i+1]
	}

	fg := autofunc.NewGradient([]*autofunc.Variable{flagsVar})
	flagRes.PropagateGradient(flagsGrad, fg)

	ctrlGrad := make(linalg.Vector, queueFlagCount+len(pushDataGrad))
	copy(ctrlGrad, fg[flagsVar])
	copy(ctrlGrad[queueFlagCount:], pushDataGrad)

	return ctrlGrad, downstream
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
	res.ControlIn = ctrl
	res.Last = q

	return &res
}

type queueUpstream struct {
	Expected  []linalg.Vector
	SizeProbs []float64
}
