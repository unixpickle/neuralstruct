// Package neuralstruct makes it possible to link recurrent
// neural networks with various data structures.
package neuralstruct

import "github.com/unixpickle/num-analysis/linalg"

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

	// Gradient computes the gradient of some value x with
	// respect to each of the input control values, given
	// the gradient of x with respect to each component of
	// the output data.
	Gradient(upstream linalg.Vector) linalg.Vector

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
	RGradient(upstream, upstreamR linalg.Vector) (grad, rgrad linalg.Vector)

	// NextRState is like NextState, but it preserves
	// and uses r-operator information.
	NextRState(control, controlR linalg.Vector) RState
}
