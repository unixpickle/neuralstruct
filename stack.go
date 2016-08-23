package neuralstruct

import (
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

const stackFlagCount = 4

const (
	stackNop int = iota
	stackPush
	stackPop
	stackReplace
)

func init() {
	s := &Stack{}
	serializer.RegisterTypedDeserializer(s.SerializerType(), DeserializeStack)
}

// Stack is a Struct which implements a stack of vectors.
type Stack struct {
	VectorSize int
}

// DeserializeStack deserializes a Stack.
func DeserializeStack(d []byte) (*Stack, error) {
	intData, err := serializer.DeserializeWithType(d)
	if err != nil {
		return nil, err
	}
	num, ok := intData.(serializer.Int)
	if !ok {
		return nil, fmt.Errorf("expected serializer.Int but got %T", intData)
	}
	return &Stack{VectorSize: int(num)}, nil
}

// ControlSize returns the number of control components,
// which varies based on the vector size.
func (s *Stack) ControlSize() int {
	return s.VectorSize + stackFlagCount
}

// DataSize returns the vector size.
func (s *Stack) DataSize() int {
	return s.VectorSize
}

// StartState returns the empty stack.
func (s *Stack) StartState() State {
	return &stackState{VecSize: s.VectorSize}
}

// StartRState returns the empty stack.
func (s *Stack) StartRState() RState {
	return &stackRState{VecSize: s.VectorSize}
}

// SerializerType returns the unique ID for serializing
// stacks with the serializer package.
func (s *Stack) SerializerType() string {
	return "github.com/unixpickle/neuralstruct.Stack"
}

// Serialize serializes the stack's parameters.
func (s *Stack) Serialize() ([]byte, error) {
	return serializer.SerializeWithType(serializer.Int(s.VectorSize))
}

type stackState struct {
	Last     *stackState
	VecSize  int
	Expected []linalg.Vector
	Control  linalg.Vector
}

func (s *stackState) Data() linalg.Vector {
	if len(s.Expected) == 0 {
		return make(linalg.Vector, s.VecSize)
	}
	return s.Expected[0]
}

func (s *stackState) Gradient(dataGrad linalg.Vector, upstreamGrad Grad) (linalg.Vector, Grad) {
	if s.Last == nil {
		panic("cannot propagate through start state")
	}

	var upstream []linalg.Vector
	if upstreamGrad != nil {
		upstream = upstreamGrad.([]linalg.Vector)
		upstream[0].Add(dataGrad)
	} else {
		upstream = make([]linalg.Vector, len(s.Expected))
		upstream[0] = dataGrad
		zeroGrad := make(linalg.Vector, len(dataGrad))
		for i := 1; i < len(upstream); i++ {
			upstream[i] = zeroGrad
		}
	}

	softmax := autofunc.Softmax{}
	flagVar := &autofunc.Variable{Vector: s.Control[:stackFlagCount]}
	flagRes := softmax.Apply(flagVar)
	flags := flagRes.Output()
	controlData := s.Control[stackFlagCount:]

	flagsDownstream := make(linalg.Vector, len(flags))
	downstream := make([]linalg.Vector, len(s.Last.Expected))

	for i, v := range s.Last.Expected {
		scaler := flags[stackNop]
		if i != 0 {
			scaler += flags[stackReplace]
		}
		downstream[i] = upstream[i].Copy().Scale(scaler)

		gradDot := v.Dot(upstream[i])
		flagsDownstream[stackNop] += gradDot
		if i != 0 {
			flagsDownstream[stackReplace] += gradDot
		}
	}

	if len(s.Last.Expected) > 0 {
		for i, v := range s.Last.Expected[1:] {
			downstream[i+1].Add(upstream[i].Copy().Scale(flags[stackPop]))
			flagsDownstream[stackPop] += upstream[i].Dot(v)
		}
	}

	controlDataDownstream := upstream[0].Copy().Scale(flags[stackPush] + flags[stackReplace])
	pushReplaceDot := upstream[0].Dot(controlData)
	flagsDownstream[stackPush] += pushReplaceDot
	flagsDownstream[stackReplace] += pushReplaceDot

	for i, v := range s.Last.Expected {
		downstream[i].Add(upstream[i+1].Copy().Scale(flags[stackPush]))
		flagsDownstream[stackPush] += upstream[i+1].Dot(v)
	}

	controlDownstream := make(linalg.Vector, len(s.Control))
	copy(controlDownstream[stackFlagCount:], controlDataDownstream)
	flagGrad := autofunc.Gradient{flagVar: controlDownstream[:stackFlagCount]}
	flagRes.PropagateGradient(flagsDownstream, flagGrad)

	return controlDownstream, downstream
}

func (s *stackState) NextState(control linalg.Vector) State {
	softmax := autofunc.Softmax{}
	flagVar := &autofunc.Variable{Vector: control[:stackFlagCount]}
	flags := softmax.Apply(flagVar).Output()
	controlData := control[stackFlagCount:]

	newState := &stackState{
		Last:     s,
		VecSize:  s.VecSize,
		Expected: make([]linalg.Vector, len(s.Expected)+1),
		Control:  control,
	}

	for i, v := range s.Expected {
		scaler := flags[stackNop]
		if i != 0 {
			scaler += flags[stackReplace]
		}
		newState.Expected[i] = v.Copy().Scale(scaler)
	}
	newState.Expected[len(s.Expected)] = make(linalg.Vector, s.VecSize)

	if len(s.Expected) > 0 {
		for i, v := range s.Expected[1:] {
			newState.Expected[i].Add(v.Copy().Scale(flags[stackPop]))
		}
	}

	newState.Expected[0].Add(controlData.Copy().Scale(flags[stackPush] + flags[stackReplace]))
	for i, v := range s.Expected {
		newState.Expected[i+1].Add(v.Copy().Scale(flags[stackPush]))
	}

	return newState
}

type stackRState struct {
	Last      *stackRState
	VecSize   int
	Expected  []linalg.Vector
	ExpectedR []linalg.Vector
	Control   linalg.Vector
	ControlR  linalg.Vector
}

func (s *stackRState) Data() linalg.Vector {
	if len(s.Expected) == 0 {
		return make(linalg.Vector, s.VecSize)
	}
	return s.Expected[0]
}

func (s *stackRState) RData() linalg.Vector {
	if len(s.ExpectedR) == 0 {
		return make(linalg.Vector, s.VecSize)
	}
	return s.ExpectedR[0]
}

func (s *stackRState) RGradient(dataGrad, dataGradR linalg.Vector,
	upstreamGrad RGrad) (linalg.Vector, linalg.Vector, RGrad) {
	if s.Last == nil {
		panic("cannot propagate through start state")
	}

	var upstream, upstreamR []linalg.Vector
	if upstreamGrad != nil {
		upstreamVal := upstreamGrad.([2][]linalg.Vector)
		upstream = upstreamVal[0]
		upstreamR = upstreamVal[1]
		upstream[0].Add(dataGrad)
		upstreamR[0].Add(dataGradR)
	} else {
		upstream = make([]linalg.Vector, len(s.Expected))
		upstreamR = make([]linalg.Vector, len(s.ExpectedR))
		upstream[0] = dataGrad
		upstreamR[0] = dataGradR
		zeroGrad := make(linalg.Vector, len(dataGrad))
		for i := 1; i < len(upstream); i++ {
			upstream[i] = zeroGrad
			upstreamR[i] = zeroGrad
		}
	}

	softmax := autofunc.Softmax{}
	flagVar := &autofunc.Variable{Vector: s.Control[:stackFlagCount]}
	flagRVar := &autofunc.RVariable{
		Variable:   flagVar,
		ROutputVec: s.ControlR[:stackFlagCount],
	}
	flagRes := softmax.ApplyR(autofunc.RVector{}, flagRVar)
	flags := flagRes.Output()
	flagsR := flagRes.ROutput()
	controlData := s.Control[stackFlagCount:]
	controlDataR := s.ControlR[stackFlagCount:]

	flagsDownstream := make(linalg.Vector, len(flags))
	downstream := make([]linalg.Vector, len(s.Last.Expected))
	flagsDownstreamR := make(linalg.Vector, len(flags))
	downstreamR := make([]linalg.Vector, len(s.Last.Expected))

	for i, v := range s.Last.Expected {
		vR := s.Last.ExpectedR[i]

		scaler := flags[stackNop]
		scalerR := flagsR[stackNop]
		if i != 0 {
			scaler += flags[stackReplace]
			scalerR += flagsR[stackReplace]
		}
		downstream[i] = upstream[i].Copy().Scale(scaler)
		downstreamR[i] = upstream[i].Copy().Scale(scalerR)
		downstreamR[i].Add(upstreamR[i].Copy().Scale(scaler))

		gradDot := v.Dot(upstream[i])
		gradDotR := vR.Dot(upstream[i]) + v.Dot(upstreamR[i])
		flagsDownstream[stackNop] += gradDot
		flagsDownstreamR[stackNop] += gradDotR
		if i != 0 {
			flagsDownstream[stackReplace] += gradDot
			flagsDownstreamR[stackReplace] += gradDotR
		}
	}

	if len(s.Last.Expected) > 0 {
		for i, v := range s.Last.Expected[1:] {
			vR := s.Last.ExpectedR[i+1]
			downstream[i+1].Add(upstream[i].Copy().Scale(flags[stackPop]))
			downstreamR[i+1].Add(upstreamR[i].Copy().Scale(flags[stackPop]))
			downstreamR[i+1].Add(upstream[i].Copy().Scale(flagsR[stackPop]))
			flagsDownstream[stackPop] += upstream[i].Dot(v)
			flagsDownstreamR[stackPop] += upstreamR[i].Dot(v) + upstream[i].Dot(vR)
		}
	}

	controlDataDownstream := upstream[0].Copy().Scale(flags[stackPush] + flags[stackReplace])
	controlDataDownstreamR := upstream[0].Copy().Scale(flagsR[stackPush] + flagsR[stackReplace])
	controlDataDownstreamR.Add(upstreamR[0].Copy().Scale(flags[stackPush] + flags[stackReplace]))
	pushReplaceDot := upstream[0].Dot(controlData)
	pushReplaceDotR := upstreamR[0].Dot(controlData) + upstream[0].Dot(controlDataR)
	flagsDownstream[stackPush] += pushReplaceDot
	flagsDownstream[stackReplace] += pushReplaceDot
	flagsDownstreamR[stackPush] += pushReplaceDotR
	flagsDownstreamR[stackReplace] += pushReplaceDotR

	for i, v := range s.Last.Expected {
		vR := s.Last.ExpectedR[i]
		downstream[i].Add(upstream[i+1].Copy().Scale(flags[stackPush]))
		downstreamR[i].Add(upstreamR[i+1].Copy().Scale(flags[stackPush]))
		downstreamR[i].Add(upstream[i+1].Copy().Scale(flagsR[stackPush]))
		flagsDownstream[stackPush] += upstream[i+1].Dot(v)
		flagsDownstreamR[stackPush] += upstreamR[i+1].Dot(v) + upstream[i+1].Dot(vR)
	}

	controlDownstream := make(linalg.Vector, len(s.Control))
	controlDownstreamR := make(linalg.Vector, len(s.Control))
	copy(controlDownstream[stackFlagCount:], controlDataDownstream)
	copy(controlDownstreamR[stackFlagCount:], controlDataDownstreamR)
	flagGrad := autofunc.Gradient{flagVar: controlDownstream[:stackFlagCount]}
	flagRGrad := autofunc.RGradient{flagVar: controlDownstreamR[:stackFlagCount]}
	flagRes.PropagateRGradient(flagsDownstream, flagsDownstreamR, flagRGrad, flagGrad)

	return controlDownstream, controlDownstreamR,
		[2][]linalg.Vector{downstream, downstreamR}
}

func (s *stackRState) NextRState(control, controlR linalg.Vector) RState {
	softmax := autofunc.Softmax{}
	flagVar := &autofunc.Variable{Vector: control[:stackFlagCount]}
	flagRVar := &autofunc.RVariable{
		Variable:   flagVar,
		ROutputVec: controlR[:stackFlagCount],
	}
	flagRes := softmax.ApplyR(autofunc.RVector{}, flagRVar)
	flags := flagRes.Output()
	flagsR := flagRes.ROutput()
	controlData := control[stackFlagCount:]
	controlDataR := controlR[stackFlagCount:]

	newState := &stackRState{
		Last:      s,
		VecSize:   s.VecSize,
		Expected:  make([]linalg.Vector, len(s.Expected)+1),
		ExpectedR: make([]linalg.Vector, len(s.Expected)+1),
		Control:   control,
		ControlR:  controlR,
	}

	for i, v := range s.Expected {
		vR := s.ExpectedR[i]

		scaler := flags[stackNop]
		scalerR := flagsR[stackNop]
		if i != 0 {
			scaler += flags[stackReplace]
			scalerR += flagsR[stackReplace]
		}

		newState.Expected[i] = v.Copy().Scale(scaler)
		newState.ExpectedR[i] = v.Copy().Scale(scalerR).Add(vR.Copy().Scale(scaler))
	}
	newState.Expected[len(s.Expected)] = make(linalg.Vector, s.VecSize)
	newState.ExpectedR[len(s.Expected)] = make(linalg.Vector, s.VecSize)

	if len(s.Expected) > 0 {
		for i, v := range s.Expected[1:] {
			vR := s.ExpectedR[i+1]
			newState.Expected[i].Add(v.Copy().Scale(flags[stackPop]))
			newState.ExpectedR[i].Add(v.Copy().Scale(flagsR[stackPop]))
			newState.ExpectedR[i].Add(vR.Copy().Scale(flags[stackPop]))
		}
	}

	newState.Expected[0].Add(controlData.Copy().Scale(flags[stackPush] + flags[stackReplace]))
	newState.ExpectedR[0].Add(controlDataR.Copy().Scale(flags[stackPush] + flags[stackReplace]))
	newState.ExpectedR[0].Add(controlData.Copy().Scale(flagsR[stackPush] + flagsR[stackReplace]))
	for i, v := range s.Expected {
		vR := s.ExpectedR[i]
		newState.Expected[i+1].Add(v.Copy().Scale(flags[stackPush]))
		newState.ExpectedR[i+1].Add(vR.Copy().Scale(flags[stackPush]))
		newState.ExpectedR[i+1].Add(v.Copy().Scale(flagsR[stackPush]))
	}

	return newState
}
