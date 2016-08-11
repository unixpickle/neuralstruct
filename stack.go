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
func (s *Stack) StartState() StructState {
	return &stackState{
		stacks: [][]linalg.Vector{[]linalg.Vector{}},
		probs:  []float64{1},
	}
}

type stackState struct {
	// TODO: optimize not to be exponential in storage.
	last    *stackState
	stacks  [][]linalg.Vector
	probs   linalg.Vector
	control linalg.Vector
	data    linalg.Vector
}

func (s *stackState) Data() linalg.Vector {
	return s.data
}

func (s *stackState) Gradient(upstream linalg.Vector) linalg.Vector {
	// TODO: this.
	return nil
}

func (s *stackState) RGradient(upstream, upstreamR linalg.Vector) (grad, rgrad linalg.Vector) {
	// TODO: this.
	return nil, nil
}

func (s *stackState) NextState(control linalg.Vector) StructState {
	softmax := autofunc.Softmax{}
	flagVar := &autofunc.Variable{Vector: control[:stackFlagCount]}
	flags := softmax.Apply(flagVar).Output()
	controlData := control[stackFlagCount:]

	newState := &stackState{
		last:    s,
		stacks:  nil,
		probs:   nil,
		control: control,
	}

	for i, stack := range s.stacks {
		prob := s.probs[i]
		newState.stacks = append(newState.stacks, stack)
		newState.probs = append(newState.probs, prob*flags[stackNop])
		newState.stacks = append(newState.stacks, pushStack(stack, controlData))
		newState.probs = append(newState.probs, prob*flags[stackPush])
		newState.stacks = append(newState.stacks, popStack(stack))
		newState.probs = append(newState.probs, prob*flags[stackPop])
		newState.stacks = append(newState.stacks, replaceStack(stack, controlData))
		newState.probs = append(newState.probs, prob*flags[stackReplace])
	}

	newState.data = newState.computeData()
	return newState
}

func (s *stackState) computeData() linalg.Vector {
	var res linalg.Vector
	for i, stack := range s.stacks {
		if len(stack) == 0 {
			continue
		}
		head := stack[len(stack)-1]
		prob := s.probs[i]
		if res == nil {
			res = head.Copy().Scale(prob)
		} else {
			res.Add(head.Copy().Scale(prob))
		}
	}
	if res == nil {
		return make(linalg.Vector, len(s.control)-stackFlagCount)
	}
	return res
}

func pushStack(s []linalg.Vector, x linalg.Vector) []linalg.Vector {
	return append(copyStack(s), x)
}

func popStack(s []linalg.Vector) []linalg.Vector {
	if len(s) == 0 {
		return nil
	}
	return s[:len(s)-1]
}

func replaceStack(s []linalg.Vector, x linalg.Vector) []linalg.Vector {
	if len(s) == 0 {
		return s
	} else {
		res := copyStack(s)
		res[len(res)-1] = x
		return res
	}
}

func copyStack(s []linalg.Vector) []linalg.Vector {
	res := make([]linalg.Vector, len(s))
	copy(res, s)
	return res
}
