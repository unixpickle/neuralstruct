package neuralstruct

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

// These are the control flags (in order) of a Stack.
const (
	StackNop int = iota
	StackPush
	StackPop
	StackReplace
)

func init() {
	s := &Stack{}
	serializer.RegisterTypedDeserializer(s.SerializerType(), DeserializeStack)
}

// Stack is a Struct which implements a stack of vectors.
type Stack struct {
	VectorSize int

	// NoReplace, if true, indicates that the Stack should
	// not provide a "replace" flag in the control signal.
	NoReplace bool

	// PushBias determines an optional bias towards pushing
	// from the SuggestedActivation() method.
	// Reasonable values are -1, 0, or 1, for pushing being
	// e times less likely, unbiased, or e times more likely.
	PushBias float64
}

// DeserializeStack deserializes a Stack.
func DeserializeStack(d []byte) (*Stack, error) {
	var res Stack
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// ControlSize returns the number of control components,
// which varies based on the vector size.
func (s *Stack) ControlSize() int {
	return s.flagCount() + s.VectorSize
}

// DataSize returns the vector size.
func (s *Stack) DataSize() int {
	return s.VectorSize
}

// StartState returns the empty stack.
func (s *Stack) StartState() State {
	return &stackState{Stack: *s}
}

// StartRState returns the empty stack.
func (s *Stack) StartRState() RState {
	return &stackRState{Stack: *s}
}

// SerializerType returns the unique ID for serializing
// stacks with the serializer package.
func (s *Stack) SerializerType() string {
	return "github.com/unixpickle/neuralstruct.Stack"
}

// Serialize serializes the stack's parameters.
func (s *Stack) Serialize() ([]byte, error) {
	return json.Marshal(s)
}

// SuggestedActivation returns an activation function
// which applies a hyperbolic tangent to the data outputs
// while leaving the control outputs untouched.
func (s *Stack) SuggestedActivation() neuralnet.Layer {
	res := &PartialActivation{
		Ranges:      []ComponentRange{{Start: s.flagCount(), End: s.ControlSize()}},
		Activations: []neuralnet.Layer{&neuralnet.HyperbolicTangent{}},
	}
	if s.PushBias != 0 {
		res.Ranges = append([]ComponentRange{{Start: StackPush, End: StackPush + 1}},
			res.Ranges...)
		res.Activations = append([]neuralnet.Layer{
			&neuralnet.RescaleLayer{Scale: 1, Bias: s.PushBias},
		}, res.Activations...)
	}
	return res
}

func (s *Stack) flagCount() int {
	if s.NoReplace {
		return 3
	} else {
		return 4
	}
}

type stackState struct {
	Last     *stackState
	Stack    Stack
	Expected []linalg.Vector
	Control  linalg.Vector
}

func (s *stackState) Data() linalg.Vector {
	if len(s.Expected) == 0 {
		return make(linalg.Vector, s.Stack.VectorSize)
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
	flagVar := &autofunc.Variable{Vector: s.Control[:s.Stack.flagCount()]}
	flagRes := softmax.Apply(flagVar)
	flags := flagRes.Output()
	controlData := s.Control[s.Stack.flagCount():]

	flagsDownstream := make(linalg.Vector, len(flags))
	downstream := make([]linalg.Vector, len(s.Last.Expected))

	for i, v := range s.Last.Expected {
		scaler := flags[StackNop]
		if i != 0 && !s.Stack.NoReplace {
			scaler += flags[StackReplace]
		}
		downstream[i] = upstream[i].Copy().Scale(scaler)

		gradDot := v.Dot(upstream[i])
		flagsDownstream[StackNop] += gradDot
		if i != 0 && !s.Stack.NoReplace {
			flagsDownstream[StackReplace] += gradDot
		}
	}

	if len(s.Last.Expected) > 0 {
		for i, v := range s.Last.Expected[1:] {
			downstream[i+1].Add(upstream[i].Copy().Scale(flags[StackPop]))
			flagsDownstream[StackPop] += upstream[i].Dot(v)
		}
	}

	pushReplaceProb := flags[StackPush]
	if !s.Stack.NoReplace {
		pushReplaceProb += flags[StackReplace]
	}
	controlDataDownstream := upstream[0].Copy().Scale(pushReplaceProb)
	pushReplaceDot := upstream[0].Dot(controlData)
	flagsDownstream[StackPush] += pushReplaceDot
	if !s.Stack.NoReplace {
		flagsDownstream[StackReplace] += pushReplaceDot
	}

	for i, v := range s.Last.Expected {
		downstream[i].Add(upstream[i+1].Copy().Scale(flags[StackPush]))
		flagsDownstream[StackPush] += upstream[i+1].Dot(v)
	}

	controlDownstream := make(linalg.Vector, len(s.Control))
	copy(controlDownstream[len(flags):], controlDataDownstream)
	flagGrad := autofunc.Gradient{flagVar: controlDownstream[:len(flags)]}
	flagRes.PropagateGradient(flagsDownstream, flagGrad)

	return controlDownstream, downstream
}

func (s *stackState) NextState(control linalg.Vector) State {
	softmax := autofunc.Softmax{}
	flagVar := &autofunc.Variable{Vector: control[:s.Stack.flagCount()]}
	flags := softmax.Apply(flagVar).Output()
	controlData := control[s.Stack.flagCount():]

	newState := &stackState{
		Last:     s,
		Stack:    s.Stack,
		Expected: make([]linalg.Vector, len(s.Expected)+1),
		Control:  control,
	}

	for i, v := range s.Expected {
		scaler := flags[StackNop]
		if i != 0 && !s.Stack.NoReplace {
			scaler += flags[StackReplace]
		}
		newState.Expected[i] = v.Copy().Scale(scaler)
	}
	newState.Expected[len(s.Expected)] = make(linalg.Vector, s.Stack.VectorSize)

	if len(s.Expected) > 0 {
		for i, v := range s.Expected[1:] {
			newState.Expected[i].Add(v.Copy().Scale(flags[StackPop]))
		}
	}

	pushReplace := flags[StackPush]
	if !s.Stack.NoReplace {
		pushReplace += flags[StackReplace]
	}
	newState.Expected[0].Add(controlData.Copy().Scale(pushReplace))
	for i, v := range s.Expected {
		newState.Expected[i+1].Add(v.Copy().Scale(flags[StackPush]))
	}

	return newState
}

type stackRState struct {
	Last      *stackRState
	Stack     Stack
	Expected  []linalg.Vector
	ExpectedR []linalg.Vector
	Control   linalg.Vector
	ControlR  linalg.Vector
}

func (s *stackRState) Data() linalg.Vector {
	if len(s.Expected) == 0 {
		return make(linalg.Vector, s.Stack.VectorSize)
	}
	return s.Expected[0]
}

func (s *stackRState) RData() linalg.Vector {
	if len(s.ExpectedR) == 0 {
		return make(linalg.Vector, s.Stack.VectorSize)
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
	flagVar := &autofunc.Variable{Vector: s.Control[:s.Stack.flagCount()]}
	flagRVar := &autofunc.RVariable{
		Variable:   flagVar,
		ROutputVec: s.ControlR[:s.Stack.flagCount()],
	}
	flagRes := softmax.ApplyR(autofunc.RVector{}, flagRVar)
	flags := flagRes.Output()
	flagsR := flagRes.ROutput()
	controlData := s.Control[s.Stack.flagCount():]
	controlDataR := s.ControlR[s.Stack.flagCount():]

	flagsDownstream := make(linalg.Vector, len(flags))
	downstream := make([]linalg.Vector, len(s.Last.Expected))
	flagsDownstreamR := make(linalg.Vector, len(flags))
	downstreamR := make([]linalg.Vector, len(s.Last.Expected))

	for i, v := range s.Last.Expected {
		vR := s.Last.ExpectedR[i]

		scaler := flags[StackNop]
		scalerR := flagsR[StackNop]
		if i != 0 && !s.Stack.NoReplace {
			scaler += flags[StackReplace]
			scalerR += flagsR[StackReplace]
		}
		downstream[i] = upstream[i].Copy().Scale(scaler)
		downstreamR[i] = upstream[i].Copy().Scale(scalerR)
		downstreamR[i].Add(upstreamR[i].Copy().Scale(scaler))

		gradDot := v.Dot(upstream[i])
		gradDotR := vR.Dot(upstream[i]) + v.Dot(upstreamR[i])
		flagsDownstream[StackNop] += gradDot
		flagsDownstreamR[StackNop] += gradDotR
		if i != 0 && !s.Stack.NoReplace {
			flagsDownstream[StackReplace] += gradDot
			flagsDownstreamR[StackReplace] += gradDotR
		}
	}

	if len(s.Last.Expected) > 0 {
		for i, v := range s.Last.Expected[1:] {
			vR := s.Last.ExpectedR[i+1]
			downstream[i+1].Add(upstream[i].Copy().Scale(flags[StackPop]))
			downstreamR[i+1].Add(upstreamR[i].Copy().Scale(flags[StackPop]))
			downstreamR[i+1].Add(upstream[i].Copy().Scale(flagsR[StackPop]))
			flagsDownstream[StackPop] += upstream[i].Dot(v)
			flagsDownstreamR[StackPop] += upstreamR[i].Dot(v) + upstream[i].Dot(vR)
		}
	}

	var controlDataDownstream, controlDataDownstreamR linalg.Vector
	if s.Stack.NoReplace {
		controlDataDownstream = upstream[0].Copy().Scale(flags[StackPush])
		controlDataDownstreamR = upstream[0].Copy().Scale(flagsR[StackPush])
		controlDataDownstreamR.Add(upstreamR[0].Copy().Scale(flags[StackPush]))
	} else {
		controlDataDownstream = upstream[0].Copy().Scale(flags[StackPush] +
			flags[StackReplace])
		controlDataDownstreamR = upstream[0].Copy().Scale(flagsR[StackPush] +
			flagsR[StackReplace])
		controlDataDownstreamR.Add(upstreamR[0].Copy().Scale(flags[StackPush] +
			flags[StackReplace]))
	}
	pushReplaceDot := upstream[0].Dot(controlData)
	pushReplaceDotR := upstreamR[0].Dot(controlData) + upstream[0].Dot(controlDataR)
	flagsDownstream[StackPush] += pushReplaceDot
	flagsDownstreamR[StackPush] += pushReplaceDotR
	if !s.Stack.NoReplace {
		flagsDownstream[StackReplace] += pushReplaceDot
		flagsDownstreamR[StackReplace] += pushReplaceDotR
	}

	for i, v := range s.Last.Expected {
		vR := s.Last.ExpectedR[i]
		downstream[i].Add(upstream[i+1].Copy().Scale(flags[StackPush]))
		downstreamR[i].Add(upstreamR[i+1].Copy().Scale(flags[StackPush]))
		downstreamR[i].Add(upstream[i+1].Copy().Scale(flagsR[StackPush]))
		flagsDownstream[StackPush] += upstream[i+1].Dot(v)
		flagsDownstreamR[StackPush] += upstreamR[i+1].Dot(v) + upstream[i+1].Dot(vR)
	}

	controlDownstream := make(linalg.Vector, len(s.Control))
	controlDownstreamR := make(linalg.Vector, len(s.Control))
	copy(controlDownstream[len(flags):], controlDataDownstream)
	copy(controlDownstreamR[len(flags):], controlDataDownstreamR)
	flagGrad := autofunc.Gradient{flagVar: controlDownstream[:len(flags)]}
	flagRGrad := autofunc.RGradient{flagVar: controlDownstreamR[:len(flags)]}
	flagRes.PropagateRGradient(flagsDownstream, flagsDownstreamR, flagRGrad, flagGrad)

	return controlDownstream, controlDownstreamR,
		[2][]linalg.Vector{downstream, downstreamR}
}

func (s *stackRState) NextRState(control, controlR linalg.Vector) RState {
	softmax := autofunc.Softmax{}
	flagVar := &autofunc.Variable{Vector: control[:s.Stack.flagCount()]}
	flagRVar := &autofunc.RVariable{
		Variable:   flagVar,
		ROutputVec: controlR[:s.Stack.flagCount()],
	}
	flagRes := softmax.ApplyR(autofunc.RVector{}, flagRVar)
	flags := flagRes.Output()
	flagsR := flagRes.ROutput()
	controlData := control[s.Stack.flagCount():]
	controlDataR := controlR[s.Stack.flagCount():]

	newState := &stackRState{
		Last:      s,
		Stack:     s.Stack,
		Expected:  make([]linalg.Vector, len(s.Expected)+1),
		ExpectedR: make([]linalg.Vector, len(s.Expected)+1),
		Control:   control,
		ControlR:  controlR,
	}

	for i, v := range s.Expected {
		vR := s.ExpectedR[i]

		scaler := flags[StackNop]
		scalerR := flagsR[StackNop]
		if i != 0 && !s.Stack.NoReplace {
			scaler += flags[StackReplace]
			scalerR += flagsR[StackReplace]
		}

		newState.Expected[i] = v.Copy().Scale(scaler)
		newState.ExpectedR[i] = v.Copy().Scale(scalerR).Add(vR.Copy().Scale(scaler))
	}
	newState.Expected[len(s.Expected)] = make(linalg.Vector, s.Stack.VectorSize)
	newState.ExpectedR[len(s.Expected)] = make(linalg.Vector, s.Stack.VectorSize)

	if len(s.Expected) > 0 {
		for i, v := range s.Expected[1:] {
			vR := s.ExpectedR[i+1]
			newState.Expected[i].Add(v.Copy().Scale(flags[StackPop]))
			newState.ExpectedR[i].Add(v.Copy().Scale(flagsR[StackPop]))
			newState.ExpectedR[i].Add(vR.Copy().Scale(flags[StackPop]))
		}
	}

	pushReplace := flags[StackPush]
	pushReplaceR := flagsR[StackPush]
	if !s.Stack.NoReplace {
		pushReplace += flags[StackReplace]
		pushReplaceR += flagsR[StackReplace]
	}
	newState.Expected[0].Add(controlData.Copy().Scale(pushReplace))
	newState.ExpectedR[0].Add(controlDataR.Copy().Scale(pushReplace))
	newState.ExpectedR[0].Add(controlData.Copy().Scale(pushReplaceR))
	for i, v := range s.Expected {
		vR := s.ExpectedR[i]
		newState.Expected[i+1].Add(v.Copy().Scale(flags[StackPush]))
		newState.ExpectedR[i+1].Add(vR.Copy().Scale(flags[StackPush]))
		newState.ExpectedR[i+1].Add(v.Copy().Scale(flagsR[StackPush]))
	}

	return newState
}
