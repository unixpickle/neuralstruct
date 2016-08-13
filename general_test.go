package neuralstruct

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type structFunc struct {
	Struct Struct
}

// Apply treats the input vector as a joined list of
// control vectors and applies each control vector to
// the previous state in order.
// It concatenates the output data at every timestep
// and returns the result.
func (s *structFunc) Apply(in autofunc.Result) autofunc.Result {
	var joinedData linalg.Vector
	var outputs []State
	state := s.Struct.StartState()
	for i := 0; i < len(in.Output()); i += s.Struct.ControlSize() {
		control := in.Output()[i : i+s.Struct.ControlSize()]
		out := state.NextState(control)
		outputs = append(outputs, out)
		joinedData = append(joinedData, out.Data()...)
	}
	return &structFuncRes{
		Outputs:    outputs,
		JoinedData: joinedData,
		Controls:   in,
	}
}

type structFuncRes struct {
	Outputs    []State
	JoinedData linalg.Vector
	Controls   autofunc.Result
}

func (s *structFuncRes) Constant(g autofunc.Gradient) bool {
	return s.Controls.Constant(g)
}

func (s *structFuncRes) Output() linalg.Vector {
	return s.JoinedData
}

func (s *structFuncRes) PropagateGradient(upstream linalg.Vector, g autofunc.Gradient) {
	if s.Controls.Constant(g) {
		return
	}

	controlGrad := make(linalg.Vector, len(s.Controls.Output()))
	sourceIdx := len(upstream)

	var stateUpstream Grad
	var destIdx int
	for t := len(s.Outputs) - 1; t >= 0; t-- {
		out := s.Outputs[t]
		upstreamPart := upstream[sourceIdx-len(out.Data()) : sourceIdx]
		sourceIdx -= len(out.Data())

		var downstreamPart linalg.Vector
		downstreamPart, stateUpstream = out.Gradient(upstreamPart, stateUpstream)
		copy(controlGrad[destIdx:], downstreamPart)
		destIdx += len(downstreamPart)
	}

	s.Controls.PropagateGradient(controlGrad, g)
}
