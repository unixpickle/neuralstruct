package neuralstruct

import (
	"math"
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
)

type stackDataOp struct {
	Nop     float64
	Push    float64
	Pop     float64
	Replace float64
	Value   linalg.Vector
}

func (s *stackDataOp) Control() linalg.Vector {
	var vec linalg.Vector
	if s.Replace == 0 {
		vec = make(linalg.Vector, len(s.Value)+3)
		copy(vec[3:], s.Value)
	} else {
		vec = make(linalg.Vector, len(s.Value)+4)
		copy(vec[4:], s.Value)
		vec[3] = math.Log(s.Replace)
	}
	vec[0] = math.Log(s.Nop)
	vec[1] = math.Log(s.Push)
	vec[2] = math.Log(s.Pop)
	return vec
}

type stackDataTest struct {
	VecSize   int
	NoReplace bool
	Ops       []stackDataOp
	Expected  []linalg.Vector
}

func TestStackData(t *testing.T) {
	tests := []stackDataTest{
		stackDataTest{
			VecSize: 4,
			Ops: []stackDataOp{
				{0.25, 0.25, 0.25, 0.25, []float64{1, 2, 3, 4}},
				{0.1, 0.3, 0.2, 0.4, []float64{4, 3, 2, 1}},
				{0.2, 0.3, 0.24, 0.26, []float64{6, 7, 8, 9}},
				{0.3, 0.2, 0.26, 0.24, []float64{-1, -2, -3, -4}},
			},
			Expected: []linalg.Vector{
				{0.5, 1, 1.5, 2},
				{4*(0.3+0.4) + 1*0.5*0.1, 3*(0.3+0.4) + 2*0.5*0.1,
					2*(0.3+0.4) + 3*0.5*0.1, 1*(0.3+0.4) + 4*0.5*0.1},
				{0.2*(4*(0.3+0.4)+1*0.5*0.1) + 1*0.24*0.5*0.3 + 6*(0.3+0.26),
					0.2*(3*(0.3+0.4)+2*0.5*0.1) + 2*0.24*0.5*0.3 + 7*(0.3+0.26),
					0.2*(2*(0.3+0.4)+3*0.5*0.1) + 3*0.24*0.5*0.3 + 8*(0.3+0.26),
					0.2*(1*(0.3+0.4)+4*0.5*0.1) + 4*0.24*0.5*0.3 + 9*(0.3+0.26)},
				{0.99004, 0.65708, 0.32412, -0.00884},
			},
		},
		stackDataTest{
			VecSize:   4,
			NoReplace: true,
			Ops: []stackDataOp{
				{0.25, 0.25, 0.25, 0, []float64{1, 2, 3, 4}},
				{0.1, 0.3, 0.2, 0, []float64{4, 3, 2, 1}},
				{0.2, 0.3, 0.24, 0, []float64{6, 7, 8, 9}},
				{0.3, 0.2, 0.26, 0, []float64{-1, -2, -3, -4}},
			},
			Expected: []linalg.Vector{
				{1.0 / 3, 2.0 / 3, 1, 4.0 / 3},
				{2.05555555555555, 1.61111111111111, 1.16666666666666, 0.72222222222222},
				{3.04204204204204, 3.38138138138138, 3.72072072072072, 4.06006006006006},
				{1.23814604077761, 1.06270744428639, 0.88726884779516, 0.71183025130393},
			},
		},
	}
	for i, test := range tests {
		stack := Stack{VectorSize: test.VecSize, NoReplace: test.NoReplace}
		state := stack.StartState()
		for j, op := range test.Ops {
			state = state.NextState(op.Control())
			exp := test.Expected[j]
			if !statesEqual(state.Data(), exp) {
				t.Error("test", i, "time", j, "expected", exp,
					"but got", state.Data())
				break
			}
		}
	}
}

func TestStackDerivatives(t *testing.T) {
	testAllDerivatives(t, &Stack{VectorSize: 4})
}

func TestStackDerivativesNoReplace(t *testing.T) {
	testAllDerivatives(t, &Stack{VectorSize: 4, NoReplace: true})
}

func BenchmarkStackForward(b *testing.B) {
	forwardBenchmark(b, &Stack{VectorSize: benchmarkVectorSize})
}

func BenchmarkStackBackward(b *testing.B) {
	backwardBenchmark(b, &Stack{VectorSize: benchmarkVectorSize})
}
