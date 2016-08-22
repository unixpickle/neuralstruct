package neuralstruct

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
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
	vec := make(linalg.Vector, len(s.Value)+4)
	copy(vec[4:], s.Value)
	vec[0] = math.Log(s.Nop)
	vec[1] = math.Log(s.Push)
	vec[2] = math.Log(s.Pop)
	vec[3] = math.Log(s.Replace)
	return vec
}

type stackDataTest struct {
	VecSize  int
	Ops      []stackDataOp
	Expected []linalg.Vector
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
	}
	for i, test := range tests {
		stack := Stack{VectorSize: test.VecSize}
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

func TestStackShallowGradient(t *testing.T) {
	testStackGradient(t, 1)
}

func TestStackDeepGradient(t *testing.T) {
	testStackGradient(t, 4)
}

func testStackGradient(t *testing.T, steps int) {
	f, ctrl := stackTestFunc()
	inVec := make(linalg.Vector, ctrl*steps)
	for i := range inVec {
		inVec[i] = rand.NormFloat64()
	}
	inVar := &autofunc.Variable{Vector: inVec}
	test := functest.FuncTest{
		F:     f,
		Vars:  []*autofunc.Variable{inVar},
		Input: inVar,
	}
	test.Run(t)
}

func TestStackShallowRGradient(t *testing.T) {
	testStackRGradient(t, 1)
}

func TestStackDeepRGradient(t *testing.T) {
	testStackRGradient(t, 4)
}

func testStackRGradient(t *testing.T, steps int) {
	f, ctrl := stackTestRFunc()
	inVec := make(linalg.Vector, steps*ctrl)
	inVecR := make(linalg.Vector, steps*ctrl)
	for i := range inVec {
		inVec[i] = rand.NormFloat64()
		inVecR[i] = rand.NormFloat64()
	}
	inVar := &autofunc.Variable{Vector: inVec}
	test := functest.RFuncTest{
		F:     f,
		Vars:  []*autofunc.Variable{inVar},
		Input: inVar,
		RV:    autofunc.RVector{inVar: inVecR},
	}
	test.Run(t)
}

func stackTestFunc() (funcOut autofunc.Func, inSize int) {
	res := &structFunc{Struct: &Stack{VectorSize: 4}}
	return res, res.Struct.ControlSize()
}

func stackTestRFunc() (funcOut autofunc.RFunc, inSize int) {
	res := &structRFunc{Struct: &Stack{VectorSize: 4}}
	return res, res.Struct.ControlSize()
}
