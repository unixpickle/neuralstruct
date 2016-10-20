package neuralstruct

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

const (
	benchmarkTimeSteps  = 100
	benchmarkVectorSize = 40
)

func statesEqual(d1, d2 linalg.Vector) bool {
	if len(d1) != len(d2) {
		return false
	}
	for i, x := range d1 {
		if math.Abs(x-d2[i]) > 1e-5 {
			return false
		}
	}
	return true
}

func forwardBenchmark(b *testing.B, s Struct) {
	inputVec := make(linalg.Vector, s.DataSize())
	for i := range inputVec {
		inputVec[i] = rand.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		state := s.StartState()
		for j := 0; j < benchmarkTimeSteps; j++ {
			state = state.NextState(inputVec)
		}
	}
}

func backwardBenchmark(b *testing.B, s Struct) {
	inputVec := make(linalg.Vector, s.ControlSize())
	for i := range inputVec {
		inputVec[i] = rand.NormFloat64()
	}
	upstreamVec := make(linalg.Vector, s.DataSize())
	for i := 0; i < b.N; i++ {
		state := s.StartState()
		var states []State
		for j := 0; j < benchmarkTimeSteps; j++ {
			state = state.NextState(inputVec)
			states = append(states, state)
		}
		var ups Grad
		for j := len(states) - 1; j >= 0; j-- {
			_, ups = states[j].Gradient(upstreamVec, ups)
		}
	}
}

func testAllDerivatives(t *testing.T, s RStruct) {
	for depth := 1; depth <= 4; depth++ {
		name := fmt.Sprintf("Depth%d", depth)
		t.Run(name, func(t *testing.T) {
			f := &structRFunc{Struct: s}
			inVec := make(linalg.Vector, depth*s.ControlSize())
			inVecR := make(linalg.Vector, depth*s.ControlSize())
			for i := range inVec {
				inVec[i] = rand.NormFloat64()
				inVecR[i] = rand.NormFloat64()
			}
			inVar := &autofunc.Variable{Vector: inVec}
			test := functest.RFuncChecker{
				F:     f,
				Vars:  []*autofunc.Variable{inVar},
				Input: inVar,
				RV:    autofunc.RVector{inVar: inVecR},
			}
			test.FullCheck(t)
		})
	}
}

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
		state = state.NextState(control)
		outputs = append(outputs, state)
		joinedData = append(joinedData, state.Data()...)
	}
	return &structFuncRes{
		Outputs:    outputs,
		JoinedData: joinedData,
		Controls:   in,
	}
}

type structRFunc struct {
	Struct RStruct
}

// Apply is like structFunc.Apply.
func (s *structRFunc) Apply(in autofunc.Result) autofunc.Result {
	f := structFunc{s.Struct}
	return f.Apply(in)
}

// ApplyR is like Apply but with r-operator support.
func (s *structRFunc) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	var joinedData, joinedRData linalg.Vector
	var outputs []RState
	state := s.Struct.StartRState()
	for i := 0; i < len(in.Output()); i += s.Struct.ControlSize() {
		control := in.Output()[i : i+s.Struct.ControlSize()]
		controlR := in.ROutput()[i : i+s.Struct.ControlSize()]
		state = state.NextRState(control, controlR)
		outputs = append(outputs, state)
		joinedData = append(joinedData, state.Data()...)
		joinedRData = append(joinedRData, state.RData()...)
	}
	return &structFuncRRes{
		Outputs:     outputs,
		JoinedData:  joinedData,
		JoinedRData: joinedRData,
		Controls:    in,
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
	destIdx := len(controlGrad) - len(controlGrad)/len(s.Outputs)

	var stateUpstream Grad
	for t := len(s.Outputs) - 1; t >= 0; t-- {
		out := s.Outputs[t]
		upstreamPart := upstream[sourceIdx-len(out.Data()) : sourceIdx]
		sourceIdx -= len(out.Data())

		var downstreamPart linalg.Vector
		downstreamPart, stateUpstream = out.Gradient(upstreamPart, stateUpstream)
		copy(controlGrad[destIdx:], downstreamPart)
		destIdx -= len(downstreamPart)
	}

	s.Controls.PropagateGradient(controlGrad, g)
}

type structFuncRRes struct {
	Outputs     []RState
	JoinedData  linalg.Vector
	JoinedRData linalg.Vector
	Controls    autofunc.RResult
}

func (s *structFuncRRes) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return s.Controls.Constant(rg, g)
}

func (s *structFuncRRes) Output() linalg.Vector {
	return s.JoinedData
}

func (s *structFuncRRes) ROutput() linalg.Vector {
	return s.JoinedRData
}

func (s *structFuncRRes) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	if s.Controls.Constant(rg, g) {
		return
	}

	controlGrad := make(linalg.Vector, len(s.Controls.Output()))
	controlGradR := make(linalg.Vector, len(s.Controls.Output()))
	sourceIdx := len(upstream)
	destIdx := len(controlGrad) - len(controlGrad)/len(s.Outputs)

	var stateUpstream RGrad
	for t := len(s.Outputs) - 1; t >= 0; t-- {
		out := s.Outputs[t]
		upstreamPart := upstream[sourceIdx-len(out.Data()) : sourceIdx]
		upstreamPartR := upstreamR[sourceIdx-len(out.Data()) : sourceIdx]
		sourceIdx -= len(out.Data())

		var downstreamPart, downstreamPartR linalg.Vector
		downstreamPart, downstreamPartR, stateUpstream = out.RGradient(upstreamPart,
			upstreamPartR, stateUpstream)
		copy(controlGrad[destIdx:], downstreamPart)
		copy(controlGradR[destIdx:], downstreamPartR)
		destIdx -= len(downstreamPart)
	}

	s.Controls.PropagateRGradient(controlGrad, controlGradR, rg, g)
}
