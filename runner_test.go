package neuralstruct

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const runnerTestSeqLen = 4

func TestRunner(t *testing.T) {
	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  5,
			OutputCount: 11,
		},
	}
	outNet.Randomize()
	block := rnn.StackedBlock{
		rnn.NewLSTM(6, 5),
		rnn.NewNetworkBlock(outNet, 0),
	}
	structure := &Stack{VectorSize: 4}

	runner := Runner{Block: block, Struct: structure}
	inputSeq := make([]autofunc.Result, runnerTestSeqLen)
	outputs := make([]linalg.Vector, runnerTestSeqLen)

	for i := range outputs {
		input := make(linalg.Vector, 2)
		for j := range input {
			input[j] = rand.NormFloat64()
		}
		outputs[i] = runner.StepTime(input)
		inputSeq[i] = &autofunc.Variable{Vector: input}
	}

	seqFunc := &SeqFunc{
		Block:  block,
		Struct: structure,
	}
	expectedOutputs := seqFunc.BatchSeqs([][]autofunc.Result{inputSeq}).OutputSeqs()[0]

	if len(expectedOutputs) != len(outputs) {
		t.Fatal("unexpected output count")
	}
	for time, expected := range expectedOutputs {
		actual := outputs[time]
		if len(expected) != len(actual) {
			t.Fatalf("time %d: expected output vector of len %d but got len %d",
				time, len(expected), len(actual))
		}
		for i, x := range expected {
			a := actual[i]
			if math.Abs(x-a) > 1e-5 {
				t.Errorf("time %d: entry %d should be %f but got %f", time, i, x, a)
			}
		}
	}
}
