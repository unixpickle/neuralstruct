package neuralstruct

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

const queueFlagCount = 3

const (
	queueNop int = iota
	queuePush
	queuePop
)

func init() {
	var q Queue
	serializer.RegisterTypedDeserializer(q.SerializerType(), DeserializeQueue)
}

// A Queue is a differentiable probabilistic FIFO queue
// of real-valued vectors.
type Queue struct {
	VectorSize int
}

// DeserializeQueue deserializes a Queue.
func DeserializeQueue(d []byte) (*Queue, error) {
	var q Queue
	if err := json.Unmarshal(d, &q); err != nil {
		return nil, err
	}
	return &q, nil
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
		SizeProbs:  []float64{1},
		OutputData: make(linalg.Vector, q.VectorSize),
	}
}

// StartRState returns a state representing an empty queue.
func (q *Queue) StartRState() RState {
	zeroVec := make(linalg.Vector, q.VectorSize)
	return &queueRState{
		SizeProbs:   []float64{1},
		RSizeProbs:  []float64{0},
		OutputData:  zeroVec,
		ROutputData: zeroVec,
	}
}

// SerializerType returns the unique ID used to serialize
// Queues with the serializer package.
func (q *Queue) SerializerType() string {
	return "github.com/unixpickle/neuralstruct.Queue"
}

// Serialize encodes the queue as binary data.
func (q *Queue) Serialize() ([]byte, error) {
	return json.Marshal(q)
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

type queueRState struct {
	Expected    []linalg.Vector
	RExpected   []linalg.Vector
	SizeProbs   []float64
	RSizeProbs  []float64
	OutputData  linalg.Vector
	ROutputData linalg.Vector

	ControlIn  linalg.Vector
	RControlIn linalg.Vector
	Last       *queueRState
}

func (q *queueRState) Data() linalg.Vector {
	return q.OutputData
}

func (q *queueRState) RData() linalg.Vector {
	return q.ROutputData
}

func (q *queueRState) RGradient(dataGrad, dataGradR linalg.Vector,
	upstreamGrad RGrad) (linalg.Vector, linalg.Vector, RGrad) {
	if q.Last == nil {
		panic("cannot propagate through start state")
	}
	softmax := autofunc.Softmax{}
	flagsVar := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: q.ControlIn[:queueFlagCount]},
		ROutputVec: q.RControlIn[:queueFlagCount],
	}
	flagRes := softmax.ApplyR(autofunc.RVector{}, flagsVar)
	flags := flagRes.Output()
	flagsR := flagRes.ROutput()

	flagsGrad := make(linalg.Vector, queueFlagCount)
	flagsGradR := make(linalg.Vector, queueFlagCount)

	var upstream *queueRUpstream
	if upstreamGrad != nil {
		upstream = upstreamGrad.(*queueRUpstream)
		upstream.Expected[0].Add(dataGrad)
		upstream.RExpected[0].Add(dataGradR)
	} else {
		upstream = new(queueRUpstream)
		upstream.SizeProbs = make([]float64, len(q.SizeProbs))
		upstream.RSizeProbs = make([]float64, len(q.SizeProbs))
		upstream.Expected = make([]linalg.Vector, len(q.Expected))
		upstream.RExpected = make([]linalg.Vector, len(q.Expected))
		upstream.Expected[0] = dataGrad
		upstream.RExpected[0] = dataGradR
		zeroVec := make(linalg.Vector, len(dataGrad))
		for i := 1; i < len(upstream.Expected); i++ {
			upstream.Expected[i] = zeroVec
			upstream.RExpected[i] = zeroVec
		}
	}

	downstream := &queueRUpstream{
		Expected:   make([]linalg.Vector, len(q.Last.Expected)),
		RExpected:  make([]linalg.Vector, len(q.Last.Expected)),
		SizeProbs:  make([]float64, len(q.Last.SizeProbs)),
		RSizeProbs: make([]float64, len(q.Last.SizeProbs)),
	}
	for i, vec := range q.Last.Expected {
		vecR := q.Last.RExpected[i]
		downstream.Expected[i] = upstream.Expected[i].Copy().Scale(flags[queueNop] +
			flags[queuePush])
		downstream.RExpected[i] = upstream.RExpected[i].Copy().Scale(flags[queueNop] +
			flags[queuePush])
		downstream.RExpected[i].Add(upstream.Expected[i].Copy().Scale(flagsR[queueNop] +
			flagsR[queuePush]))
		vecDot := vec.Dot(upstream.Expected[i])
		vecDotR := vec.Dot(upstream.RExpected[i]) + vecR.Dot(upstream.Expected[i])
		flagsGrad[queueNop] += vecDot
		flagsGrad[queuePush] += vecDot
		flagsGradR[queueNop] += vecDotR
		flagsGradR[queuePush] += vecDotR
		if i > 0 {
			downstream.Expected[i].Add(upstream.Expected[i-1].Copy().Scale(flags[queuePop]))
			downstream.RExpected[i].Add(upstream.RExpected[i-1].Copy().Scale(flags[queuePop]))
			downstream.RExpected[i].Add(upstream.Expected[i-1].Copy().Scale(flagsR[queuePop]))
			flagsGrad[queuePop] += vec.Dot(upstream.Expected[i-1])
			flagsGradR[queuePop] += vecR.Dot(upstream.Expected[i-1]) +
				vec.Dot(upstream.RExpected[i-1])
		}
	}

	pushDataGrad := make(linalg.Vector, len(q.Data()))
	pushDataGradR := make(linalg.Vector, len(q.RData()))

	for i, prob := range q.Last.SizeProbs {
		probR := q.Last.RSizeProbs[i]
		pushData := q.ControlIn[queueFlagCount:]
		pushDataR := q.RControlIn[queueFlagCount:]
		pushDataGrad.Add(upstream.Expected[i].Copy().Scale(flags[queuePush] * prob))
		pushDataGradR.Add(upstream.RExpected[i].Copy().Scale(flags[queuePush] * prob))
		pushDataGradR.Add(upstream.Expected[i].Copy().Scale(flagsR[queuePush]*prob +
			flags[queuePush]*probR))
		upstreamDot := upstream.Expected[i].Dot(pushData)
		upstreamDotR := upstream.RExpected[i].Dot(pushData) + upstream.Expected[i].Dot(pushDataR)
		flagsGrad[queuePush] += prob * upstreamDot
		flagsGradR[queuePush] += probR*upstreamDot + prob*upstreamDotR
		downstream.SizeProbs[i] += flags[queuePush] * upstreamDot
		downstream.RSizeProbs[i] += flagsR[queuePush]*upstreamDot +
			flags[queuePush]*upstreamDotR
	}

	for i, old := range q.Last.SizeProbs {
		oldR := q.Last.RSizeProbs[i]
		downstream.SizeProbs[i] += flags[queueNop] * upstream.SizeProbs[i]
		downstream.RSizeProbs[i] += flagsR[queueNop]*upstream.SizeProbs[i] +
			flags[queueNop]*upstream.RSizeProbs[i]
		flagsGrad[queueNop] += old * upstream.SizeProbs[i]
		flagsGradR[queueNop] += oldR*upstream.SizeProbs[i] + old*upstream.RSizeProbs[i]
		if i > 0 {
			downstream.SizeProbs[i] += flags[queuePop] * upstream.SizeProbs[i-1]
			downstream.RSizeProbs[i] += flagsR[queuePop]*upstream.SizeProbs[i-1] +
				flags[queuePop]*upstream.RSizeProbs[i-1]
			flagsGrad[queuePop] += old * upstream.SizeProbs[i-1]
			flagsGradR[queuePop] += oldR*upstream.SizeProbs[i-1] +
				old*upstream.RSizeProbs[i-1]
		} else {
			downstream.SizeProbs[i] += flags[queuePop] * upstream.SizeProbs[i]
			downstream.RSizeProbs[i] += flagsR[queuePop]*upstream.SizeProbs[i] +
				flags[queuePop]*upstream.RSizeProbs[i]
			flagsGrad[queuePop] += old * upstream.SizeProbs[i]
			flagsGradR[queuePop] += oldR*upstream.SizeProbs[i] + old*upstream.RSizeProbs[i]
		}
		downstream.SizeProbs[i] += flags[queuePush] * upstream.SizeProbs[i+1]
		downstream.RSizeProbs[i] += flagsR[queuePush]*upstream.SizeProbs[i+1] +
			flags[queuePush]*upstream.RSizeProbs[i+1]
		flagsGrad[queuePush] += old * upstream.SizeProbs[i+1]
		flagsGradR[queuePush] += oldR*upstream.SizeProbs[i+1] + old*upstream.RSizeProbs[i+1]
	}

	fg := autofunc.NewGradient([]*autofunc.Variable{flagsVar.Variable})
	fgR := autofunc.NewRGradient([]*autofunc.Variable{flagsVar.Variable})
	flagRes.PropagateRGradient(flagsGrad, flagsGradR, fgR, fg)

	ctrlGrad := make(linalg.Vector, queueFlagCount+len(pushDataGrad))
	copy(ctrlGrad, fg[flagsVar.Variable])
	copy(ctrlGrad[queueFlagCount:], pushDataGrad)
	ctrlGradR := make(linalg.Vector, queueFlagCount+len(pushDataGrad))
	copy(ctrlGradR, fgR[flagsVar.Variable])
	copy(ctrlGradR[queueFlagCount:], pushDataGradR)

	return ctrlGrad, ctrlGradR, downstream
}

func (q *queueRState) NextRState(ctrl, ctrlR linalg.Vector) RState {
	probs := ctrl[:queueFlagCount]
	softmax := autofunc.Softmax{}
	probsVar := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: probs},
		ROutputVec: ctrlR[:queueFlagCount],
	}
	flagsRes := softmax.ApplyR(autofunc.RVector{}, probsVar)
	flags := flagsRes.Output()
	flagsR := flagsRes.ROutput()

	var res queueRState

	res.Expected = make([]linalg.Vector, len(q.Expected)+1)
	res.RExpected = make([]linalg.Vector, len(q.Expected)+1)
	res.SizeProbs = make([]float64, len(q.SizeProbs)+1)
	res.RSizeProbs = make([]float64, len(q.SizeProbs)+1)

	for i, old := range q.SizeProbs {
		oldR := q.RSizeProbs[i]
		res.SizeProbs[i] += old * flags[queueNop]
		res.RSizeProbs[i] += oldR*flags[queueNop] + old*flagsR[queueNop]
		if i > 0 {
			res.SizeProbs[i-1] += old * flags[queuePop]
			res.RSizeProbs[i-1] += oldR*flags[queuePop] + old*flagsR[queuePop]
		} else {
			res.SizeProbs[i] += old * flags[queuePop]
			res.RSizeProbs[i] += oldR*flags[queuePop] + old*flagsR[queuePop]
		}
		res.SizeProbs[i+1] += old * flags[queuePush]
		res.RSizeProbs[i+1] += oldR*flags[queuePush] + old*flagsR[queuePush]
	}

	for i, vec := range q.Expected {
		vecR := q.RExpected[i]
		res.Expected[i] = vec.Copy().Scale(flags[queueNop] + flags[queuePush])
		res.RExpected[i] = vec.Copy().Scale(flagsR[queueNop] + flagsR[queuePush])
		res.RExpected[i].Add(vecR.Copy().Scale(flags[queueNop] + flags[queuePush]))
		if i > 0 {
			res.Expected[i-1].Add(vec.Copy().Scale(flags[queuePop]))
			res.RExpected[i-1].Add(vec.Copy().Scale(flagsR[queuePop]))
			res.RExpected[i-1].Add(vecR.Copy().Scale(flags[queuePop]))
		}
	}

	pushData := ctrl[queueFlagCount:]
	pushDataR := ctrlR[queueFlagCount:]
	for i, prob := range q.SizeProbs {
		probR := q.RSizeProbs[i]
		pushVec := pushData.Copy().Scale(flags[queuePush] * prob)
		pushVecR := pushDataR.Copy().Scale(flags[queuePush] * prob)
		pushVecR.Add(pushData.Copy().Scale(flagsR[queuePush]*prob + flags[queuePush]*probR))
		if i == len(res.Expected)-1 {
			res.Expected[i] = pushVec
			res.RExpected[i] = pushVecR
		} else {
			res.Expected[i].Add(pushVec)
			res.RExpected[i].Add(pushVecR)
		}
	}

	res.OutputData = res.Expected[0]
	res.ROutputData = res.RExpected[0]
	res.ControlIn = ctrl
	res.RControlIn = ctrlR
	res.Last = q

	return &res
}

type queueRUpstream struct {
	Expected   []linalg.Vector
	RExpected  []linalg.Vector
	SizeProbs  []float64
	RSizeProbs []float64
}
