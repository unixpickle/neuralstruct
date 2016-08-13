package neuralstruct

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const stackFlagCount = 4

const (
	stackNop int = iota
	stackPush
	stackPop
	stackReplace
)

// Stack is a Struct which implements a stack of vectors.
type Stack struct {
	VectorSize int
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
	return &stackState{vecSize: s.VectorSize}
}

// StartRState returns the empty stack.
func (s *Stack) StartRState() RState {
	return &stackRState{vecSize: s.VectorSize}
}

type stackState struct {
	last     *stackState
	vecSize  int
	expected []linalg.Vector
	control  linalg.Vector
}

func (s *stackState) Data() linalg.Vector {
	if len(s.expected) == 0 {
		return make(linalg.Vector, s.vecSize)
	}
	return s.expected[0]
}

func (s *stackState) Gradient(dataGrad linalg.Vector, upstreamGrad Grad) (linalg.Vector, Grad) {
	if s.last == nil {
		panic("cannot propagate through start state")
	}

	var upstream []linalg.Vector
	if upstreamGrad != nil {
		upstream = upstreamGrad.([]linalg.Vector)
		upstream[0].Add(dataGrad)
	} else {
		upstream = make([]linalg.Vector, len(s.expected))
		upstream[0] = dataGrad
		zeroGrad := make(linalg.Vector, len(dataGrad))
		for i := 1; i < len(upstream); i++ {
			upstream[i] = zeroGrad
		}
	}

	softmax := autofunc.Softmax{}
	flagVar := &autofunc.Variable{Vector: s.control[:stackFlagCount]}
	flagRes := softmax.Apply(flagVar)
	flags := flagRes.Output()
	controlData := s.control[stackFlagCount:]

	flagsDownstream := make(linalg.Vector, len(flags))
	downstream := make([]linalg.Vector, len(s.last.expected))

	for i, v := range s.last.expected {
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

	if len(s.last.expected) > 0 {
		for i, v := range s.last.expected[1:] {
			downstream[i+1].Add(upstream[i].Copy().Scale(flags[stackPop]))
			flagsDownstream[stackPop] += upstream[i].Dot(v)
		}
	}

	controlDataDownstream := upstream[0].Copy().Scale(flags[stackPush] + flags[stackReplace])
	pushReplaceDot := upstream[0].Dot(controlData)
	flagsDownstream[stackPush] += pushReplaceDot
	flagsDownstream[stackReplace] += pushReplaceDot

	for i, v := range s.last.expected {
		downstream[i].Add(upstream[i+1].Copy().Scale(flags[stackPush]))
		flagsDownstream[stackPush] += upstream[i+1].Dot(v)
	}

	controlDownstream := make(linalg.Vector, len(s.control))
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
		last:     s,
		vecSize:  s.vecSize,
		expected: make([]linalg.Vector, len(s.expected)+1),
		control:  control,
	}

	for i, v := range s.expected {
		scaler := flags[stackNop]
		if i != 0 {
			scaler += flags[stackReplace]
		}
		newState.expected[i] = v.Copy().Scale(scaler)
	}
	newState.expected[len(s.expected)] = make(linalg.Vector, s.vecSize)

	if len(s.expected) > 0 {
		for i, v := range s.expected[1:] {
			newState.expected[i].Add(v.Copy().Scale(flags[stackPop]))
		}
	}

	newState.expected[0].Add(controlData.Copy().Scale(flags[stackPush] + flags[stackReplace]))
	for i, v := range s.expected {
		newState.expected[i+1].Add(v.Copy().Scale(flags[stackPush]))
	}

	return newState
}

type stackRState struct {
	last      *stackRState
	vecSize   int
	expected  []linalg.Vector
	expectedR []linalg.Vector
	control   linalg.Vector
	controlR  linalg.Vector
}

func (s *stackRState) Data() linalg.Vector {
	if len(s.expected) == 0 {
		return make(linalg.Vector, s.vecSize)
	}
	return s.expected[0]
}

func (s *stackRState) RData() linalg.Vector {
	if len(s.expectedR) == 0 {
		return make(linalg.Vector, s.vecSize)
	}
	return s.expectedR[0]
}

func (s *stackRState) RGradient(dataGrad, dataGradR linalg.Vector,
	upstreamGrad RGrad) (linalg.Vector, linalg.Vector, RGrad) {
	// TODO: this.
	zeroVec := make(linalg.Vector, 4+s.vecSize)
	return zeroVec, zeroVec, nil
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
		last:      s,
		vecSize:   s.vecSize,
		expected:  make([]linalg.Vector, len(s.expected)+1),
		expectedR: make([]linalg.Vector, len(s.expected)+1),
		control:   control,
		controlR:  controlR,
	}

	for i, v := range s.expected {
		vR := s.expectedR[i]

		scaler := flags[stackNop]
		scalerR := flagsR[stackNop]
		if i != 0 {
			scaler += flags[stackReplace]
			scalerR += flagsR[stackReplace]
		}

		newState.expected[i] = v.Copy().Scale(scaler)
		newState.expectedR[i] = v.Copy().Scale(scalerR).Add(vR.Copy().Scale(scaler))
	}
	newState.expected[len(s.expected)] = make(linalg.Vector, s.vecSize)
	newState.expectedR[len(s.expected)] = make(linalg.Vector, s.vecSize)

	if len(s.expected) > 0 {
		for i, v := range s.expected[1:] {
			vR := s.expectedR[i+1]
			newState.expected[i].Add(v.Copy().Scale(flags[stackPop]))
			newState.expectedR[i].Add(v.Copy().Scale(flagsR[stackPop]))
			newState.expectedR[i].Add(vR.Copy().Scale(flags[stackPop]))
		}
	}

	newState.expected[0].Add(controlData.Copy().Scale(flags[stackPush] + flags[stackReplace]))
	newState.expectedR[0].Add(controlDataR.Copy().Scale(flags[stackPush] + flags[stackReplace]))
	newState.expectedR[0].Add(controlData.Copy().Scale(flagsR[stackPush] + flagsR[stackReplace]))
	for i, v := range s.expected {
		vR := s.expectedR[i]
		newState.expected[i+1].Add(v.Copy().Scale(flags[stackPush]))
		newState.expectedR[i+1].Add(vR.Copy().Scale(flags[stackPush]))
		newState.expectedR[i+1].Add(v.Copy().Scale(flagsR[stackPush]))
	}

	return newState
}
