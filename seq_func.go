package neuralstruct

import (
	"errors"
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn"
)

// SeqFunc is an rnn.SeqFunc which wraps an rnn.Block
// and gives the block access to an RStruct.
type SeqFunc struct {
	Block  rnn.Block
	Struct RStruct
}

// DeserializeSeqFunc deserializes a SeqFunc.
func DeserializeSeqFunc(d []byte) (*SeqFunc, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 2 {
		return nil, errors.New("invalid SeqFunc slice")
	}
	block, ok1 := slice[0].(rnn.Block)
	structure, ok2 := slice[1].(RStruct)
	if !ok1 || !ok2 {
		return nil, errors.New("invalid SeqFunc slice")
	}
	return &SeqFunc{Block: block, Struct: structure}, nil
}

// BatchSeqs applies s to a list of sequences.
func (s *SeqFunc) BatchSeqs(seqs [][]autofunc.Result) rnn.ResultSeqs {
	var res seqFuncOutput
	res.PackedOut = make([][]linalg.Vector, len(seqs))

	zeroStateVec := make(linalg.Vector, s.Block.StateSize())

	var t int
	for {
		step := &seqFuncOutputStep{
			InStateVars: make([]*autofunc.Variable, len(seqs)),
			InputVars:   make([]*autofunc.Variable, len(seqs)),
			Inputs:      make([]autofunc.Result, len(seqs)),
			LaneToOut:   map[int]int{},
		}
		var input rnn.BlockInput
		var inputStates []State
		for l, seq := range seqs {
			if len(seq) <= t {
				continue
			}
			step.LaneToOut[l] = len(input.Inputs)
			step.Inputs[l] = seq[t]
			var lastState State
			var lastStateVec linalg.Vector
			if t > 0 {
				last := res.Steps[t-1]
				lastState = last.OutStates[last.LaneToOut[l]]
				lastStateVec = last.Outputs.States()[last.LaneToOut[l]]
			} else {
				lastState = s.Struct.StartState()
				lastStateVec = zeroStateVec
			}
			inputStates = append(inputStates, lastState)
			inVec := make(linalg.Vector, len(lastState.Data())+len(seq[t].Output()))
			copy(inVec, lastState.Data())
			copy(inVec[len(lastState.Data()):], seq[t].Output())
			step.InputVars[l] = &autofunc.Variable{Vector: inVec}
			step.InStateVars[l] = &autofunc.Variable{Vector: lastStateVec}
			input.Inputs = append(input.Inputs, step.InputVars[l])
			input.States = append(input.States, step.InStateVars[l])
		}
		if len(step.LaneToOut) == 0 {
			break
		}
		step.Outputs = s.Block.Batch(&input)
		for i, inState := range inputStates {
			outVec := step.Outputs.Outputs()[i]
			ctrl := outVec[:s.Struct.ControlSize()]
			step.OutStates = append(step.OutStates, inState.NextState(ctrl))
		}
		res.Steps = append(res.Steps, step)
		for l, outIdx := range step.LaneToOut {
			outVec := step.Outputs.Outputs()[outIdx]
			outData := outVec[s.Struct.ControlSize():]
			res.PackedOut[l] = append(res.PackedOut[l], outData)
		}
		t++
	}

	return &res
}

// BatchSeqsR applies s to a list of sequences.
func (s *SeqFunc) BatchSeqsR(rv autofunc.RVector, seqs [][]autofunc.RResult) rnn.RResultSeqs {
	var res seqFuncROutput
	res.PackedOut = make([][]linalg.Vector, len(seqs))
	res.RPackedOut = make([][]linalg.Vector, len(seqs))

	zeroStateVec := make(linalg.Vector, s.Block.StateSize())

	var t int
	for {
		step := &seqFuncROutputStep{
			InStateVars: make([]*autofunc.RVariable, len(seqs)),
			InputVars:   make([]*autofunc.RVariable, len(seqs)),
			Inputs:      make([]autofunc.RResult, len(seqs)),
			LaneToOut:   map[int]int{},
		}
		var input rnn.BlockRInput
		var inputStates []RState
		for l, seq := range seqs {
			if len(seq) <= t {
				continue
			}
			step.LaneToOut[l] = len(input.Inputs)
			step.Inputs[l] = seq[t]
			var lastState RState
			var lastStateVec linalg.Vector
			var lastStateVecR linalg.Vector
			if t > 0 {
				last := res.Steps[t-1]
				lastState = last.OutStates[last.LaneToOut[l]]
				lastStateVec = last.Outputs.States()[last.LaneToOut[l]]
				lastStateVecR = last.Outputs.RStates()[last.LaneToOut[l]]
			} else {
				lastState = s.Struct.StartRState()
				lastStateVec = zeroStateVec
				lastStateVecR = zeroStateVec
			}
			inputStates = append(inputStates, lastState)

			inVec := make(linalg.Vector, len(lastState.Data())+len(seq[t].Output()))
			copy(inVec, lastState.Data())
			copy(inVec[len(lastState.Data()):], seq[t].Output())

			inVecR := make(linalg.Vector, len(lastState.RData())+len(seq[t].ROutput()))
			copy(inVecR, lastState.RData())
			copy(inVecR[len(lastState.RData()):], seq[t].ROutput())

			step.InputVars[l] = &autofunc.RVariable{
				Variable:   &autofunc.Variable{Vector: inVec},
				ROutputVec: inVecR,
			}
			step.InStateVars[l] = &autofunc.RVariable{
				Variable:   &autofunc.Variable{Vector: lastStateVec},
				ROutputVec: lastStateVecR,
			}
			input.Inputs = append(input.Inputs, step.InputVars[l])
			input.States = append(input.States, step.InStateVars[l])
		}
		if len(step.LaneToOut) == 0 {
			break
		}
		step.Outputs = s.Block.BatchR(rv, &input)
		for i, inState := range inputStates {
			outVec := step.Outputs.Outputs()[i]
			outVecR := step.Outputs.ROutputs()[i]
			ctrl := outVec[:s.Struct.ControlSize()]
			ctrlR := outVecR[:s.Struct.ControlSize()]
			step.OutStates = append(step.OutStates, inState.NextRState(ctrl, ctrlR))
		}
		res.Steps = append(res.Steps, step)
		for l, outIdx := range step.LaneToOut {
			outVec := step.Outputs.Outputs()[outIdx]
			outVecR := step.Outputs.ROutputs()[outIdx]
			outData := outVec[s.Struct.ControlSize():]
			outDataR := outVecR[s.Struct.ControlSize():]
			res.PackedOut[l] = append(res.PackedOut[l], outData)
			res.RPackedOut[l] = append(res.RPackedOut[l], outDataR)
		}
		t++
	}

	return &res
}

// Parameters returns the underlying block's parameters
// if it implements sgd.Learner, or nil otherwise.
func (s *SeqFunc) Parameters() []*autofunc.Variable {
	if l, ok := s.Block.(sgd.Learner); ok {
		return l.Parameters()
	} else {
		return nil
	}
}

// SerializerType returns the unique ID used to serialize
// SeqFuncs with the serializer package.
func (r *SeqFunc) SerializerType() string {
	return "github.com/unixpickle/neuralstruct.SeqFunc"
}

// Serialize serializes the underlying block and struct.
// If either of those two things is not a
// serializer.Serializer, this returns an error
// a serializer.Serializer (and fails otherwise).
func (s *SeqFunc) Serialize() ([]byte, error) {
	blockSerializer, ok := s.Block.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("block is not a Serializer: %T", s.Block)
	}
	structSerializer, ok := s.Struct.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("struct is not a Serializer: %T", s.Struct)
	}
	list := []serializer.Serializer{blockSerializer, structSerializer}
	return serializer.SerializeSlice(list)
}

type seqFuncOutputStep struct {
	// These three variables always have len equal to
	// the number of lanes (some entries may be nil).
	InStateVars []*autofunc.Variable
	InputVars   []*autofunc.Variable
	Inputs      []autofunc.Result

	Outputs   rnn.BlockOutput
	OutStates []State

	// LaneToOut maps lane indices to indices in Outputs.
	LaneToOut map[int]int
}

type seqFuncOutput struct {
	Steps     []*seqFuncOutputStep
	PackedOut [][]linalg.Vector
}

func (s *seqFuncOutput) OutputSeqs() [][]linalg.Vector {
	return s.PackedOut
}

func (s *seqFuncOutput) Gradient(upstream [][]linalg.Vector, g autofunc.Gradient) {
	numLanes := len(s.PackedOut)
	if len(upstream) != numLanes {
		panic("incorrect upstream dimensions")
	}
	for i, x := range upstream {
		if len(x) != len(s.PackedOut[i]) {
			panic("incorrect upstream dimensions")
		}
	}

	stateVecUpstreams := make([]linalg.Vector, numLanes)
	stateUpstreams := make([]Grad, numLanes)
	stateDataUpstreams := make([]linalg.Vector, numLanes)

	for t := len(s.Steps) - 1; t >= 0; t-- {
		step := s.Steps[t]

		var blockUpstream rnn.UpstreamGradient
		loopUsedLanes(step.LaneToOut, func(l int) {
			outState := step.OutStates[len(blockUpstream.Outputs)]
			dataUpstream := stateDataUpstreams[l]
			if dataUpstream == nil {
				dataUpstream = make(linalg.Vector, len(outState.Data()))
			}
			stateUpstream := stateUpstreams[l]
			ctrlUpstream, gradDownstream := outState.Gradient(dataUpstream, stateUpstream)

			stateUpstreams[l] = gradDownstream

			outputUpstream := upstream[l][t]
			blockOutGrad := make(linalg.Vector, len(ctrlUpstream)+len(outputUpstream))
			copy(blockOutGrad, ctrlUpstream)
			copy(blockOutGrad[len(ctrlUpstream):], outputUpstream)
			blockUpstream.Outputs = append(blockUpstream.Outputs, blockOutGrad)

			stateVar := step.InStateVars[l]
			stateVecUpstream := stateVecUpstreams[l]
			if stateVecUpstream == nil {
				stateVecUpstream = make(linalg.Vector, len(stateVar.Vector))
			}
			blockUpstream.States = append(blockUpstream.States, stateVecUpstream)
			if t > 0 {
				g[stateVar] = make(linalg.Vector, len(stateVar.Vector))
			}
			if t > 0 || !step.Inputs[l].Constant(g) {
				g[step.InputVars[l]] = make(linalg.Vector, len(step.InputVars[l].Vector))
			}
		})

		step.Outputs.Gradient(&blockUpstream, g)

		var stateIdx int
		loopUsedLanes(step.LaneToOut, func(l int) {
			if t > 0 {
				stateVar := step.InStateVars[l]
				stateVecUpstreams[l] = g[stateVar]
				delete(g, stateVar)
			}

			if input := step.Inputs[l]; t > 0 || !input.Constant(g) {
				upstream := g[step.InputVars[l]]
				delete(g, step.InputVars[l])

				// Technically we want the length of the previous state's
				// data, but all states have the same data length.
				structDataSize := len(step.OutStates[stateIdx].Data())

				stateDataUpstreams[l] = upstream[:structDataSize]

				if !input.Constant(g) {
					input.PropagateGradient(upstream[structDataSize:], g)
				}
			}

			stateIdx++
		})
	}
}

type seqFuncROutputStep struct {
	// These three variables always have len equal to
	// the number of lanes (some entries may be nil).
	InStateVars []*autofunc.RVariable
	InputVars   []*autofunc.RVariable
	Inputs      []autofunc.RResult

	Outputs   rnn.BlockROutput
	OutStates []RState

	// LaneToOut maps lane indices to indices in Outputs.
	LaneToOut map[int]int
}

type seqFuncROutput struct {
	Steps      []*seqFuncROutputStep
	PackedOut  [][]linalg.Vector
	RPackedOut [][]linalg.Vector
}

func (s *seqFuncROutput) OutputSeqs() [][]linalg.Vector {
	return s.PackedOut
}

func (s *seqFuncROutput) ROutputSeqs() [][]linalg.Vector {
	return s.RPackedOut
}

func (s *seqFuncROutput) RGradient(upstream, upstreamR [][]linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	// TODO: this.
}

func loopUsedLanes(laneToOut map[int]int, f func(int)) {
	var lane int
	k := len(laneToOut)
	for k > 0 {
		if _, ok := laneToOut[lane]; ok {
			f(lane)
			k--
		}
		lane++
	}
}
