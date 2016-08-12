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

func (s *stackState) Gradient(upstream linalg.Vector) linalg.Vector {
	// TODO: this.
	return make(linalg.Vector, len(s.control))
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
		control:  s.control,
	}

	for i, v := range s.expected {
		newState.expected[i] = make(linalg.Vector, len(v))
		copy(newState.expected[i], s.expected[i])
		scaler := flags[stackNop]
		if i != 0 {
			scaler += flags[stackReplace]
		}
		newState.expected[i].Scale(scaler)
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
