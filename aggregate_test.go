package neuralstruct

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

func TestAggregateShallowGradient(t *testing.T) {
	testAggregateGradient(t, 1)
}

func TestAggregateDeepGradient(t *testing.T) {
	testAggregateGradient(t, 4)
}

func testAggregateGradient(t *testing.T, steps int) {
	f, ctrl := aggregateTestFunc()
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

func TestAggregateShallowRGradient(t *testing.T) {
	testAggregateRGradient(t, 1)
}

func TestAggregateDeepRGradient(t *testing.T) {
	testAggregateRGradient(t, 4)
}

func testAggregateRGradient(t *testing.T, steps int) {
	f, ctrl := aggregateTestRFunc()
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

func aggregateTestFunc() (funcOut autofunc.Func, inSize int) {
	structure := &Aggregate{
		&Stack{VectorSize: 3},
		&Queue{VectorSize: 4},
	}
	res := &structFunc{Struct: structure}
	return res, res.Struct.ControlSize()
}

func aggregateTestRFunc() (funcOut autofunc.RFunc, inSize int) {
	structure := &RAggregate{
		&Stack{VectorSize: 3},
		&Queue{VectorSize: 4},
	}
	res := &structRFunc{Struct: structure}
	return res, res.Struct.ControlSize()
}
