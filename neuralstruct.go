// Package neuralstruct makes it possible to link recurrent
// neural networks with various data structures.
package neuralstruct

// A Struct is an instance of a differentiable data structure.
type Struct interface {
	ControlSize() int
	DataSize() int
}

// A StructState is the output of a Struct at a timestep.
type StructState interface {
	// Data is the data output at this timestep.
	Data() linalg.Vector

	// Gradient computes the gradient of some value x with
	// respect to each of the input control values, given
	// the gradient of x with respect to each component of
	// the output data.
	Gradient(upstream linalg.Vector) linalg.Vector

	// RGradient is like Gradient, but it computes both the
	// gradient and the r-gradient.
	RGradient(upstream, upstreamR linalg.Vector) (grad, rgrad linalg.Vector)

	// NextState computes the next state after applying the
	// given control vector to this state.
	NextState(control linalg.Vector)
}
