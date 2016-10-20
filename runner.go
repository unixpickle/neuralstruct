package neuralstruct

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

// A Runner evaluates an rnn.Block which has been
// given control over a Struct.
type Runner struct {
	Block  rnn.Block
	Struct Struct

	curBlockState  rnn.State
	curStructState State
}

// Reset resets the current state, starting a new
// input sequence.
func (r *Runner) Reset() {
	r.curStructState = nil
	r.curBlockState = nil
}

// StepTime gives an input vector to the RNN in the
// current state and returns the RNN's output.
// This updates the Runner's internal state, meaning
// the next StepTime works off of the state caused by
// this StepTime.
func (r *Runner) StepTime(input linalg.Vector) linalg.Vector {
	res := r.StepTimeFull(input)
	return res[r.Struct.ControlSize():]
}

// StepTimeFull is like StepTime, but instead of returning
// part of the block's output, it returns the entire
// output (including the control data).
func (r *Runner) StepTimeFull(input linalg.Vector) linalg.Vector {
	if r.curBlockState == nil {
		r.curBlockState = r.Block.StartState()
		r.curStructState = r.Struct.StartState()
	}
	augmentedIn := make(linalg.Vector, len(r.curStructState.Data())+len(input))
	copy(augmentedIn, r.curStructState.Data())
	copy(augmentedIn[len(r.curStructState.Data()):], input)

	inRes := []autofunc.Result{&autofunc.Variable{Vector: augmentedIn}}
	out := r.Block.ApplyBlock([]rnn.State{r.curBlockState}, inRes)

	ctrl := out.Outputs()[0][:r.Struct.ControlSize()]

	r.curStructState = r.curStructState.NextState(ctrl)
	r.curBlockState = out.States()[0]

	return out.Outputs()[0]
}

// RunAll applies the RNN to a batch of sequences.
// It does not affect the state used by StepTime.
func (r *Runner) RunAll(seqs [][]linalg.Vector) [][]linalg.Vector {
	constIn := seqfunc.ConstResult(seqs)
	sf := &rnn.BlockSeqFunc{B: &Block{Block: r.Block, Struct: &nopRStruct{r.Struct}}}
	return sf.ApplySeqs(constIn).OutputSeqs()
}

type nopRStruct struct {
	Struct
}

func (n *nopRStruct) StartRState() RState {
	panic("StartRState not available")
}
