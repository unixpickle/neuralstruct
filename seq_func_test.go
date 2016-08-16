package neuralstruct

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

func TestSeqFuncGradient(t *testing.T) {
	f := newSeqFuncTestFunc()
	inVec := f.RandomInput()
	inVar := &autofunc.Variable{Vector: inVec}
	test := functest.FuncTest{
		F:     f,
		Vars:  append([]*autofunc.Variable{inVar}, f.Parameters()...),
		Input: inVar,
	}
	test.Run(t)
}

func TestSeqFuncRGradient(t *testing.T) {
	f := newSeqFuncTestFunc()
	inVec := f.RandomInput()
	inVecR := f.RandomInput()
	inVar := &autofunc.Variable{Vector: inVec}
	test := functest.RFuncTest{
		F:     f,
		Vars:  append([]*autofunc.Variable{inVar}, f.Parameters()...),
		Input: inVar,
		RV:    autofunc.RVector{inVar: inVecR},
	}
	test.Run(t)
}

type seqFuncTestFunc struct {
	SeqFunc rnn.SeqFunc
	SeqLens []int
	InSize  int
}

func newSeqFuncTestFunc() *seqFuncTestFunc {
	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  5,
			OutputCount: 11,
		},
	}
	outNet.Randomize()
	seqFunc := &SeqFunc{
		Block: rnn.StackedBlock{
			rnn.NewLSTM(6, 5),
			rnn.NewNetworkBlock(outNet, 0),
		},
		Struct: &Stack{VectorSize: 4},
	}
	return &seqFuncTestFunc{
		SeqFunc: seqFunc,
		SeqLens: []int{2, 1, 3, 5},
		InSize:  2,
	}
}

func (s *seqFuncTestFunc) RandomInput() linalg.Vector {
	var res linalg.Vector
	for _, size := range s.SeqLens {
		for i := 0; i < size*s.InSize; i++ {
			res = append(res, rand.NormFloat64())
		}
	}
	return res
}

func (s *seqFuncTestFunc) Parameters() []*autofunc.Variable {
	if learner, ok := s.SeqFunc.(sgd.Learner); ok {
		return learner.Parameters()
	}
	return nil
}

func (s *seqFuncTestFunc) Apply(in autofunc.Result) autofunc.Result {
	inSeqs := make([][]autofunc.Result, len(s.SeqLens))
	var idx int
	for seqIdx, seqSize := range s.SeqLens {
		seq := make([]autofunc.Result, seqSize)
		for inIdx := range seq {
			seq[inIdx] = autofunc.Slice(in, idx, idx+s.InSize)
			idx += s.InSize
		}
		inSeqs[seqIdx] = seq
	}
	output := s.SeqFunc.BatchSeqs(inSeqs)
	var joined linalg.Vector
	for _, outSeq := range output.OutputSeqs() {
		for _, outVec := range outSeq {
			joined = append(joined, outVec...)
		}
	}
	return &seqFuncTestFuncRes{
		Input:     in,
		SeqOut:    output,
		JoinedOut: joined,
	}
}

func (s *seqFuncTestFunc) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	inSeqs := make([][]autofunc.RResult, len(s.SeqLens))
	var idx int
	for seqIdx, seqSize := range s.SeqLens {
		seq := make([]autofunc.RResult, seqSize)
		for inIdx := range seq {
			seq[inIdx] = autofunc.SliceR(in, idx, idx+s.InSize)
			idx += s.InSize
		}
		inSeqs[seqIdx] = seq
	}
	output := s.SeqFunc.BatchSeqsR(rv, inSeqs)
	var joined, joinedR linalg.Vector
	for i, outSeq := range output.OutputSeqs() {
		for j, outVec := range outSeq {
			joined = append(joined, outVec...)
			joinedR = append(joined, output.ROutputSeqs()[i][j]...)
		}
	}
	return &seqFuncTestFuncRRes{
		Input:      in,
		SeqOut:     output,
		JoinedOut:  joined,
		RJoinedOut: joinedR,
	}
}

type seqFuncTestFuncRes struct {
	Input     autofunc.Result
	SeqOut    rnn.ResultSeqs
	JoinedOut linalg.Vector
}

func (s *seqFuncTestFuncRes) Output() linalg.Vector {
	return s.JoinedOut
}

func (s *seqFuncTestFuncRes) Constant(g autofunc.Gradient) bool {
	return s.Input.Constant(g)
}

func (s *seqFuncTestFuncRes) PropagateGradient(upstream linalg.Vector, g autofunc.Gradient) {
	if s.Constant(g) {
		return
	}
	splitUpstream := make([][]linalg.Vector, len(s.SeqOut.OutputSeqs()))
	for i, outSeq := range s.SeqOut.OutputSeqs() {
		splitUpstream[i] = make([]linalg.Vector, len(outSeq))
		for j, outVec := range outSeq {
			splitUpstream[i][j] = upstream[:len(outVec)]
			upstream = upstream[len(outVec):]
		}
	}
	s.SeqOut.Gradient(splitUpstream, g)
}

type seqFuncTestFuncRRes struct {
	Input      autofunc.RResult
	SeqOut     rnn.RResultSeqs
	JoinedOut  linalg.Vector
	RJoinedOut linalg.Vector
}

func (s *seqFuncTestFuncRRes) Output() linalg.Vector {
	return s.JoinedOut
}

func (s *seqFuncTestFuncRRes) ROutput() linalg.Vector {
	return s.RJoinedOut
}

func (s *seqFuncTestFuncRRes) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return s.Input.Constant(rg, g)
}

func (s *seqFuncTestFuncRRes) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	if s.Constant(rg, g) {
		return
	}
	splitUpstream := make([][]linalg.Vector, len(s.SeqOut.OutputSeqs()))
	splitUpstreamR := make([][]linalg.Vector, len(s.SeqOut.OutputSeqs()))
	for i, outSeq := range s.SeqOut.OutputSeqs() {
		splitUpstream[i] = make([]linalg.Vector, len(outSeq))
		splitUpstreamR[i] = make([]linalg.Vector, len(outSeq))
		for j, outVec := range outSeq {
			splitUpstream[i][j] = upstream[:len(outVec)]
			splitUpstreamR[i][j] = upstreamR[:len(outVec)]
			upstream = upstream[len(outVec):]
			upstreamR = upstreamR[len(outVec):]
		}
	}
	s.SeqOut.RGradient(splitUpstream, splitUpstreamR, rg, g)
}
