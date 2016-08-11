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
			},
			Expected: []linalg.Vector{
				{0.25, 0.5, 0.75, 1},
				{4*(0.3+0.4*0.25) + 1*0.25*0.1, 3*(0.3+0.4*0.25) + 2*0.25*0.1,
					2*(0.3+0.4*0.25) + 3*0.25*0.1, 1*(0.3+0.4*0.25) + 4*0.25*0.1},
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
