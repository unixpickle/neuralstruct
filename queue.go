package neuralstruct

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

const queueFlagCount = 3

// These are the control flags (in order) of a Queue.
const (
	QueueNop int = iota
	QueuePush
	QueuePop
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

// SuggestedActivation returns an activation function
// which applies a hyperbolic tangent to the data outputs
// while leaving the control outputs untouched.
func (q *Queue) SuggestedActivation() neuralnet.Layer {
	return &PartialActivation{
		Ranges:      []ComponentRange{{Start: queueFlagCount, End: q.ControlSize()}},
		Activations: []neuralnet.Layer{&neuralnet.HyperbolicTangent{}},
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
		downstream.Expected[i] = upstream.Expected[i].Copy().Scale(flags[QueueNop] +
			flags[QueuePush])
		flagsGrad[QueueNop] += vec.Dot(upstream.Expected[i])
		flagsGrad[QueuePush] += vec.Dot(upstream.Expected[i])
		if i > 0 {
			downstream.Expected[i].Add(upstream.Expected[i-1].Copy().Scale(flags[QueuePop]))
			flagsGrad[QueuePop] += vec.Dot(upstream.Expected[i-1])
		}
	}

	pushDataGrad := make(linalg.Vector, len(q.Data()))

	for i, prob := range q.Last.SizeProbs {
		pushData := q.ControlIn[queueFlagCount:]
		pushDataGrad.Add(upstream.Expected[i].Copy().Scale(flags[QueuePush] * prob))
		upstreamDot := upstream.Expected[i].Dot(pushData)
		flagsGrad[QueuePush] += prob * upstreamDot
		downstream.SizeProbs[i] += flags[QueuePush] * upstreamDot
	}

	for i, old := range q.Last.SizeProbs {
		downstream.SizeProbs[i] += flags[QueueNop] * upstream.SizeProbs[i]
		flagsGrad[QueueNop] += old * upstream.SizeProbs[i]
		if i > 0 {
			downstream.SizeProbs[i] += flags[QueuePop] * upstream.SizeProbs[i-1]
			flagsGrad[QueuePop] += old * upstream.SizeProbs[i-1]
		} else {
			downstream.SizeProbs[i] += flags[QueuePop] * upstream.SizeProbs[i]
			flagsGrad[QueuePop] += old * upstream.SizeProbs[i]
		}
		downstream.SizeProbs[i] += flags[QueuePush] * upstream.SizeProbs[i+1]
		flagsGrad[QueuePush] += old * upstream.SizeProbs[i+1]
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
		res.SizeProbs[i] += old * flags[QueueNop]
		if i > 0 {
			res.SizeProbs[i-1] += old * flags[QueuePop]
		} else {
			res.SizeProbs[i] += old * flags[QueuePop]
		}
		res.SizeProbs[i+1] += old * flags[QueuePush]
	}

	for i, vec := range q.Expected {
		res.Expected[i] = vec.Copy().Scale(flags[QueueNop] + flags[QueuePush])
		if i > 0 {
			res.Expected[i-1].Add(vec.Copy().Scale(flags[QueuePop]))
		}
	}

	pushData := ctrl[queueFlagCount:]
	for i, prob := range q.SizeProbs {
		pushVec := pushData.Copy().Scale(flags[QueuePush] * prob)
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
		downstream.Expected[i] = upstream.Expected[i].Copy().Scale(flags[QueueNop] +
			flags[QueuePush])
		downstream.RExpected[i] = upstream.RExpected[i].Copy().Scale(flags[QueueNop] +
			flags[QueuePush])
		downstream.RExpected[i].Add(upstream.Expected[i].Copy().Scale(flagsR[QueueNop] +
			flagsR[QueuePush]))
		vecDot := vec.Dot(upstream.Expected[i])
		vecDotR := vec.Dot(upstream.RExpected[i]) + vecR.Dot(upstream.Expected[i])
		flagsGrad[QueueNop] += vecDot
		flagsGrad[QueuePush] += vecDot
		flagsGradR[QueueNop] += vecDotR
		flagsGradR[QueuePush] += vecDotR
		if i > 0 {
			downstream.Expected[i].Add(upstream.Expected[i-1].Copy().Scale(flags[QueuePop]))
			downstream.RExpected[i].Add(upstream.RExpected[i-1].Copy().Scale(flags[QueuePop]))
			downstream.RExpected[i].Add(upstream.Expected[i-1].Copy().Scale(flagsR[QueuePop]))
			flagsGrad[QueuePop] += vec.Dot(upstream.Expected[i-1])
			flagsGradR[QueuePop] += vecR.Dot(upstream.Expected[i-1]) +
				vec.Dot(upstream.RExpected[i-1])
		}
	}

	pushDataGrad := make(linalg.Vector, len(q.Data()))
	pushDataGradR := make(linalg.Vector, len(q.RData()))

	for i, prob := range q.Last.SizeProbs {
		probR := q.Last.RSizeProbs[i]
		pushData := q.ControlIn[queueFlagCount:]
		pushDataR := q.RControlIn[queueFlagCount:]
		pushDataGrad.Add(upstream.Expected[i].Copy().Scale(flags[QueuePush] * prob))
		pushDataGradR.Add(upstream.RExpected[i].Copy().Scale(flags[QueuePush] * prob))
		pushDataGradR.Add(upstream.Expected[i].Copy().Scale(flagsR[QueuePush]*prob +
			flags[QueuePush]*probR))
		upstreamDot := upstream.Expected[i].Dot(pushData)
		upstreamDotR := upstream.RExpected[i].Dot(pushData) + upstream.Expected[i].Dot(pushDataR)
		flagsGrad[QueuePush] += prob * upstreamDot
		flagsGradR[QueuePush] += probR*upstreamDot + prob*upstreamDotR
		downstream.SizeProbs[i] += flags[QueuePush] * upstreamDot
		downstream.RSizeProbs[i] += flagsR[QueuePush]*upstreamDot +
			flags[QueuePush]*upstreamDotR
	}

	for i, old := range q.Last.SizeProbs {
		oldR := q.Last.RSizeProbs[i]
		downstream.SizeProbs[i] += flags[QueueNop] * upstream.SizeProbs[i]
		downstream.RSizeProbs[i] += flagsR[QueueNop]*upstream.SizeProbs[i] +
			flags[QueueNop]*upstream.RSizeProbs[i]
		flagsGrad[QueueNop] += old * upstream.SizeProbs[i]
		flagsGradR[QueueNop] += oldR*upstream.SizeProbs[i] + old*upstream.RSizeProbs[i]
		if i > 0 {
			downstream.SizeProbs[i] += flags[QueuePop] * upstream.SizeProbs[i-1]
			downstream.RSizeProbs[i] += flagsR[QueuePop]*upstream.SizeProbs[i-1] +
				flags[QueuePop]*upstream.RSizeProbs[i-1]
			flagsGrad[QueuePop] += old * upstream.SizeProbs[i-1]
			flagsGradR[QueuePop] += oldR*upstream.SizeProbs[i-1] +
				old*upstream.RSizeProbs[i-1]
		} else {
			downstream.SizeProbs[i] += flags[QueuePop] * upstream.SizeProbs[i]
			downstream.RSizeProbs[i] += flagsR[QueuePop]*upstream.SizeProbs[i] +
				flags[QueuePop]*upstream.RSizeProbs[i]
			flagsGrad[QueuePop] += old * upstream.SizeProbs[i]
			flagsGradR[QueuePop] += oldR*upstream.SizeProbs[i] + old*upstream.RSizeProbs[i]
		}
		downstream.SizeProbs[i] += flags[QueuePush] * upstream.SizeProbs[i+1]
		downstream.RSizeProbs[i] += flagsR[QueuePush]*upstream.SizeProbs[i+1] +
			flags[QueuePush]*upstream.RSizeProbs[i+1]
		flagsGrad[QueuePush] += old * upstream.SizeProbs[i+1]
		flagsGradR[QueuePush] += oldR*upstream.SizeProbs[i+1] + old*upstream.RSizeProbs[i+1]
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
		res.SizeProbs[i] += old * flags[QueueNop]
		res.RSizeProbs[i] += oldR*flags[QueueNop] + old*flagsR[QueueNop]
		if i > 0 {
			res.SizeProbs[i-1] += old * flags[QueuePop]
			res.RSizeProbs[i-1] += oldR*flags[QueuePop] + old*flagsR[QueuePop]
		} else {
			res.SizeProbs[i] += old * flags[QueuePop]
			res.RSizeProbs[i] += oldR*flags[QueuePop] + old*flagsR[QueuePop]
		}
		res.SizeProbs[i+1] += old * flags[QueuePush]
		res.RSizeProbs[i+1] += oldR*flags[QueuePush] + old*flagsR[QueuePush]
	}

	for i, vec := range q.Expected {
		vecR := q.RExpected[i]
		res.Expected[i] = vec.Copy().Scale(flags[QueueNop] + flags[QueuePush])
		res.RExpected[i] = vec.Copy().Scale(flagsR[QueueNop] + flagsR[QueuePush])
		res.RExpected[i].Add(vecR.Copy().Scale(flags[QueueNop] + flags[QueuePush]))
		if i > 0 {
			res.Expected[i-1].Add(vec.Copy().Scale(flags[QueuePop]))
			res.RExpected[i-1].Add(vec.Copy().Scale(flagsR[QueuePop]))
			res.RExpected[i-1].Add(vecR.Copy().Scale(flags[QueuePop]))
		}
	}

	pushData := ctrl[queueFlagCount:]
	pushDataR := ctrlR[queueFlagCount:]
	for i, prob := range q.SizeProbs {
		probR := q.RSizeProbs[i]
		pushVec := pushData.Copy().Scale(flags[QueuePush] * prob)
		pushVecR := pushDataR.Copy().Scale(flags[QueuePush] * prob)
		pushVecR.Add(pushData.Copy().Scale(flagsR[QueuePush]*prob + flags[QueuePush]*probR))
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
