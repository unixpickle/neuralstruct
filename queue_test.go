package neuralstruct

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

func TestQueueData(t *testing.T) {
	queue := &Queue{VectorSize: 4}
	state := queue.StartState()

	expected := [][]float64{
		{0, 0, 0, 0},
		{0.2, 0.4, 0.6, 0.8},
		{0.7*0.2 + 0.8*0.5*4, 0.7*0.4 + 0.8*0.5*3, 0.7*0.6 + 0.8*0.5*2, 0.7*0.8 + 0.8*0.5*1},
		{(0.2*0.7*0.6)*1 + (0.8*0.5*0.6+0.2*0.5*0.4)*4 + (0.8*0.2+0.3)*0.4*3,
			(0.2*0.7*0.6)*2 + (0.8*0.5*0.6+0.2*0.5*0.4)*3 + (0.8*0.2+0.3)*0.4*3,
			(0.2*0.7*0.6)*3 + (0.8*0.5*0.6+0.2*0.5*0.4)*2 + (0.8*0.2+0.3)*0.4*3,
			(0.2*0.7*0.6)*4 + (0.8*0.5*0.6+0.2*0.5*0.4)*1 + (0.8*0.2+0.3)*0.4*3},
		{1.40826, 1.2983, 1.394, 0.7778},
	}

	controls := [][]float64{
		{math.Log(0.3), math.Log(0.2), math.Log(0.5), 1, 2, 3, 4},
		{math.Log(0.2), math.Log(0.5), math.Log(0.3), 4, 3, 2, 1},
		{math.Log(0.2), math.Log(0.4), math.Log(0.4), 3, 3, 3, 3},
		{math.Log(0.25), math.Log(0.35), math.Log(0.4), 0.3, 0.5, 2, -1},
	}

	for i, exp := range expected {
		if !statesEqual(state.Data(), exp) {
			t.Errorf("bad state %d: expected %v got %v", i, exp, state.Data())
		}
		if i < len(controls) {
			state = state.NextState(controls[i])
		}
	}
}

func TestQueueShallowGradient(t *testing.T) {
	testQueueGradient(t, 1)
}

func TestQueueDeepGradient(t *testing.T) {
	testQueueGradient(t, 4)
}

func testQueueGradient(t *testing.T, steps int) {
	f, ctrl := queueTestFunc()
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

func TestQueueShallowRGradient(t *testing.T) {
	testQueueRGradient(t, 1)
}

func TestQueueDeepRGradient(t *testing.T) {
	testQueueRGradient(t, 4)
}

func testQueueRGradient(t *testing.T, steps int) {
	f, ctrl := queueTestRFunc()
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

func queueTestFunc() (funcOut autofunc.Func, inSize int) {
	res := &structFunc{Struct: &Queue{VectorSize: 4}}
	return res, res.Struct.ControlSize()
}

func queueTestRFunc() (funcOut autofunc.RFunc, inSize int) {
	res := &structRFunc{Struct: &Queue{VectorSize: 4}}
	return res, res.Struct.ControlSize()
}

func BenchmarkQueueForward(b *testing.B) {
	forwardBenchmark(b, &Queue{VectorSize: benchmarkVectorSize})
}

func BenchmarkQueueBackward(b *testing.B) {
	backwardBenchmark(b, &Queue{VectorSize: benchmarkVectorSize})
}
