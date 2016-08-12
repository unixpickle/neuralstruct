package neuralstruct

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type structFunc struct {
	StartState State
}

// Apply treats the input vector as a control vector and
// applies that control vector to s.StartState.
func (s *structFunc) Apply(in autofunc.Result) autofunc.Result {
	return &structFuncRes{
		OutState: s.StartState.NextState(in.Output()),
		Control:  in,
	}
}

type structFuncRes struct {
	OutState State
	Control  autofunc.Result
}

func (s *structFuncRes) Constant(g autofunc.Gradient) bool {
	return s.Control.Constant(g)
}

func (s *structFuncRes) Output() linalg.Vector {
	return s.OutState.Data()
}

func (s *structFuncRes) PropagateGradient(upstream linalg.Vector, g autofunc.Gradient) {
	downstream := s.OutState.Gradient(upstream)
	s.Control.PropagateGradient(downstream, g)
}
