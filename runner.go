package neuralstruct

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

// A Runner evaluates an rnn.Block which has been
// given control over a Struct.
type Runner struct {
	Block  rnn.Block
	Struct Struct

	curStateVec linalg.Vector
	curState    State
}

// Reset resets the current state, starting a new
// input sequence.
func (r *Runner) Reset() {
	r.curStateVec = nil
	r.curState = nil
}

// StepTime gives an input vector to the RNN in the
// current state and returns the RNN's output.
// This updates the Runner's internal state, meaning
// the next StepTime works off of the state caused by
// this StepTime.
func (r *Runner) StepTime(input linalg.Vector) linalg.Vector {
	if r.curStateVec == nil {
		r.curStateVec = make(linalg.Vector, r.Block.StateSize())
		r.curState = r.Struct.StartState()
	}
	augmentedIn := make(linalg.Vector, len(r.curState.Data())+len(input))
	copy(augmentedIn, r.curState.Data())
	copy(augmentedIn[len(r.curState.Data()):], input)

	blockIn := &rnn.BlockInput{
		Inputs: []*autofunc.Variable{&autofunc.Variable{Vector: augmentedIn}},
		States: []*autofunc.Variable{&autofunc.Variable{Vector: r.curStateVec}},
	}
	out := r.Block.Batch(blockIn)

	ctrl := out.Outputs()[0][:r.Struct.ControlSize()]
	outVec := out.Outputs()[0][r.Struct.ControlSize():]

	r.curState = r.curState.NextState(ctrl)
	r.curStateVec = out.States()[0]

	return outVec
}

// RunAll applies the RNN to a batch of sequences.
// It does not affect the state used by StepTime.
func (r *Runner) RunAll(seqs [][]linalg.Vector) [][]linalg.Vector {
	seqReses := make([][]autofunc.Result, len(seqs))
	for lane, seq := range seqs {
		seqReses[lane] = make([]autofunc.Result, len(seq))
		for t, v := range seq {
			seqReses[lane][t] = &autofunc.Variable{Vector: v}
		}
	}
	sf := SeqFunc{Block: r.Block, Struct: &nopRStruct{r.Struct}}
	output := sf.BatchSeqs(seqReses)
	return output.OutputSeqs()
}

type nopRStruct struct {
	Struct
}

func (n *nopRStruct) StartRState() RState {
	panic("StartRState not available")
}
