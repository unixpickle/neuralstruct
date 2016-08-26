package neuralstruct

import (
	"math"
	"testing"
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

func TestQueueDerivatives(t *testing.T) {
	testAllDerivatives(t, &Queue{VectorSize: 4})
}

func BenchmarkQueueForward(b *testing.B) {
	forwardBenchmark(b, &Queue{VectorSize: benchmarkVectorSize})
}

func BenchmarkQueueBackward(b *testing.B) {
	backwardBenchmark(b, &Queue{VectorSize: benchmarkVectorSize})
}
