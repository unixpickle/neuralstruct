// Package neuralstruct makes it possible to link recurrent
// neural networks with various data structures.
package neuralstruct

import "github.com/unixpickle/num-analysis/linalg"

// A Grad is an upstream gradient for a State.
// It is a state-specific piece of data which represents
// how much some objective function changes with respect
// to internal parameters of the state.
type Grad interface{}

// An RGrad is like a Grad, but with r-operator info.
type RGrad interface{}

// A Struct is an instance of a differentiable data structure.
type Struct interface {
	ControlSize() int
	DataSize() int
	StartState() State
}

// An RStruct is a Struct which can create RStates as well as
// plain States.
type RStruct interface {
	Struct

	StartRState() RState
}

// A State is the output of a Struct at a timestep.
type State interface {
	// Data is the data output at this timestep.
	Data() linalg.Vector

	// Gradient propagates a gradient through this state.
	// It takes the gradient of the data, as well as an
	// optional Grad for the next state.
	// If upstream is nil, it is assumed that the next
	// state has no effect on the gradient's variable.
	//
	// The upstream Grad should not be used again after
	// this call, since the object's memory may be used
	// as scratch or reused as the return value.
	Gradient(dataGrad linalg.Vector, upstream Grad) (linalg.Vector, Grad)

	// NextState computes the next state after applying the
	// given control vector to this state.
	NextState(control linalg.Vector) State
}

// An RState is like a State, but with support for the
// r-operator.
type RState interface {
	State

	// RData returns the r-operator of the state's data.
	RData() linalg.Vector

	// RGradient is like Gradient, but it computes both the
	// gradient and the r-gradient.
	//
	// The upstream RGrad should not be used again after
	// this call.
	RGradient(dataGrad, dataGradR linalg.Vector, upstream RGrad) (grad, rgrad linalg.Vector,
		downstream RGrad)

	// NextRState is like NextState, but it preserves
	// and uses r-operator information.
	NextRState(control, controlR linalg.Vector) RState
}
