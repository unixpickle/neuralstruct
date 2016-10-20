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

type blockState struct {
	BlockState  rnn.State
	StructState State
}

type blockRState struct {
	BlockState  rnn.RState
	StructState RState
}

type blockStateGrad struct {
	BlockGrad  rnn.StateGrad
	StructGrad Grad
	DataGrad   linalg.Vector
}

type blockRStateGrad struct {
	BlockGrad  rnn.RStateGrad
	StructGrad RGrad
	DataGrad   linalg.Vector
	DataGradR  linalg.Vector
}

// A Block is an rnn.Block which wraps an rnn.Block and
// gives it access to an RStruct.
type Block struct {
	Block  rnn.Block
	Struct RStruct
}

// DeserializeBlock deserializes a Block.
func DeserializeBlock(d []byte) (*Block, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 2 {
		return nil, errors.New("invalid Block slice")
	}
	block, ok1 := slice[0].(rnn.Block)
	structure, ok2 := slice[1].(RStruct)
	if !ok1 || !ok2 {
		return nil, errors.New("invalid Block slice")
	}
	return &Block{Block: block, Struct: structure}, nil
}

// StartState returns a state encompassing the block's
// start state and the struct's start state.
func (b *Block) StartState() rnn.State {
	return blockState{
		BlockState:  b.Block.StartState(),
		StructState: b.Struct.StartState(),
	}
}

// StartRState is like StartState for rnn.RStates.
func (b *Block) StartRState(rv autofunc.RVector) rnn.RState {
	return blockRState{
		BlockState:  b.Block.StartRState(rv),
		StructState: b.Struct.StartRState(),
	}
}

// PropagateStart back-propagates through the start state.
func (b *Block) PropagateStart(s []rnn.State, u []rnn.StateGrad, g autofunc.Gradient) {
	block := make([]rnn.StateGrad, len(s))
	blockS := make([]rnn.State, len(s))
	for i, stateObj := range u {
		block[i] = stateObj.(blockStateGrad).BlockGrad
		blockS[i] = s[i].(blockState).BlockState
	}
	b.Block.PropagateStart(blockS, block, g)
}

// PropagateStartR is like PropagateStart but with support
// for the r-operator.
func (b *Block) PropagateStartR(s []rnn.RState, u []rnn.RStateGrad, rg autofunc.RGradient,
	g autofunc.Gradient) {
	block := make([]rnn.RStateGrad, len(s))
	blockS := make([]rnn.RState, len(s))
	for i, stateObj := range u {
		block[i] = stateObj.(blockRStateGrad).BlockGrad
		blockS[i] = s[i].(blockRState).BlockState
	}
	b.Block.PropagateStartR(blockS, block, rg, g)
}

// ApplyBlock applies the block to a batch of inputs.
func (b *Block) ApplyBlock(s []rnn.State, in []autofunc.Result) rnn.BlockResult {
	var augmentedPool []*autofunc.Variable
	var augmentedRes []autofunc.Result
	var innerStates []rnn.State
	for i, stateObj := range s {
		state := stateObj.(blockState)
		innerStates = append(innerStates, state.BlockState)

		augLen := b.Struct.DataSize() + len(in[i].Output())
		poolVar := &autofunc.Variable{
			Vector: make(linalg.Vector, augLen),
		}
		copy(poolVar.Vector, state.StructState.Data())
		copy(poolVar.Vector[b.Struct.DataSize():], in[i].Output())

		augmentedPool = append(augmentedPool, poolVar)
		augmentedRes = append(augmentedRes, poolVar)
	}

	blockOuts := b.Block.ApplyBlock(innerStates, augmentedRes)

	var outputs []linalg.Vector
	var newStates []rnn.State

	for i, fullOut := range blockOuts.Outputs() {
		ctrlVec := fullOut[:b.Struct.ControlSize()]
		outputs = append(outputs, fullOut[b.Struct.ControlSize():])
		oldStructState := s[i].(blockState).StructState
		newStructState := oldStructState.NextState(ctrlVec)
		newStates = append(newStates, blockState{
			StructState: newStructState,
			BlockState:  blockOuts.States()[i],
		})
	}

	return &blockResult{
		Struct:    b.Struct,
		Input:     in,
		InPool:    augmentedPool,
		BlockRes:  blockOuts,
		OutStates: newStates,
		OutVecs:   outputs,
	}
}

// ApplyBlockR is like ApplyBlock, but with support for
// the r-operator.
func (b *Block) ApplyBlockR(rv autofunc.RVector, s []rnn.RState,
	in []autofunc.RResult) rnn.BlockRResult {
	var augmentedPool []*autofunc.Variable
	var augmentedRes []autofunc.RResult
	var innerStates []rnn.RState
	for i, stateObj := range s {
		state := stateObj.(blockRState)
		innerStates = append(innerStates, state.BlockState)

		augLen := b.Struct.DataSize() + len(in[i].Output())
		poolVar := &autofunc.Variable{
			Vector: make(linalg.Vector, augLen),
		}
		copy(poolVar.Vector, state.StructState.Data())
		copy(poolVar.Vector[b.Struct.DataSize():], in[i].Output())

		augR := make(linalg.Vector, len(poolVar.Vector))
		copy(augR, state.StructState.RData())
		copy(augR[b.Struct.DataSize():], in[i].ROutput())

		augmentedPool = append(augmentedPool, poolVar)
		augmentedRes = append(augmentedRes, &autofunc.RVariable{
			Variable:   poolVar,
			ROutputVec: augR,
		})
	}

	blockOuts := b.Block.ApplyBlockR(rv, innerStates, augmentedRes)

	var outputs []linalg.Vector
	var outputsR []linalg.Vector
	var newStates []rnn.RState

	rOuts := blockOuts.ROutputs()
	for i, fullOut := range blockOuts.Outputs() {
		fullOutR := rOuts[i]
		ctrlVec := fullOut[:b.Struct.ControlSize()]
		ctrlVecR := fullOutR[:b.Struct.ControlSize()]
		outputs = append(outputs, fullOut[b.Struct.ControlSize():])
		outputsR = append(outputsR, fullOutR[b.Struct.ControlSize():])
		oldStructState := s[i].(blockRState).StructState
		newStructState := oldStructState.NextRState(ctrlVec, ctrlVecR)
		newStates = append(newStates, blockRState{
			StructState: newStructState,
			BlockState:  blockOuts.RStates()[i],
		})
	}

	return &blockRResult{
		Struct:    b.Struct,
		Input:     in,
		InPool:    augmentedPool,
		BlockRes:  blockOuts,
		OutStates: newStates,
		OutVecs:   outputs,
		ROutVecs:  outputsR,
	}
}

// Parameters returns the underlying block's parameters
// if it implements sgd.Learner, or nil otherwise.
func (b *Block) Parameters() []*autofunc.Variable {
	if l, ok := b.Block.(sgd.Learner); ok {
		return l.Parameters()
	} else {
		return nil
	}
}

// SerializerType returns the unique ID used to serialize
// Blocks with the serializer package.
func (b *Block) SerializerType() string {
	return "github.com/unixpickle/neuralstruct.Block"
}

// Serialize serializes the underlying block and struct.
// If either of those two things is not a
// serializer.Serializer, this returns an error.
func (b *Block) Serialize() ([]byte, error) {
	blockSerializer, ok := b.Block.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("block is not a Serializer: %T", b.Block)
	}
	structSerializer, ok := b.Struct.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("struct is not a Serializer: %T", b.Struct)
	}
	list := []serializer.Serializer{blockSerializer, structSerializer}
	return serializer.SerializeSlice(list)
}

type blockResult struct {
	Struct    Struct
	Input     []autofunc.Result
	InPool    []*autofunc.Variable
	BlockRes  rnn.BlockResult
	OutStates []rnn.State
	OutVecs   []linalg.Vector
}

func (b *blockResult) Outputs() []linalg.Vector {
	return b.OutVecs
}

func (b *blockResult) States() []rnn.State {
	return b.OutStates
}

func (b *blockResult) PropagateGradient(u []linalg.Vector, s []rnn.StateGrad,
	g autofunc.Gradient) []rnn.StateGrad {
	if len(b.OutVecs) == 0 {
		return nil
	}

	blockUpstream := make([]linalg.Vector, len(b.OutVecs))
	blockStateUp := make([]rnn.StateGrad, len(b.OutVecs))
	structGrads := make([]Grad, len(b.OutVecs))
	for i, outState := range b.OutStates {
		blockUpstream[i] = make(linalg.Vector, len(b.OutVecs[i])+b.Struct.ControlSize())
		if s != nil && s[i] != nil {
			bsg := s[i].(blockStateGrad)
			structState := outState.(blockState).StructState
			var ctrl linalg.Vector
			ctrl, structGrads[i] = structState.Gradient(bsg.DataGrad, bsg.StructGrad)
			copy(blockUpstream[i], ctrl)
			blockStateUp[i] = bsg.BlockGrad
		}
		if u != nil {
			copy(blockUpstream[i][b.Struct.ControlSize():], u[i])
		}
	}

	for _, v := range b.InPool {
		g[v] = make(linalg.Vector, len(v.Vector))
	}
	blockDown := b.BlockRes.PropagateGradient(blockUpstream, blockStateUp, g)

	inputGrads := make([]linalg.Vector, len(b.OutVecs))
	for i, v := range b.InPool {
		inputGrads[i] = g[v]
		delete(g, v)
	}

	downstream := make([]rnn.StateGrad, len(b.OutVecs))
	for i, v := range inputGrads {
		b.Input[i].PropagateGradient(v[b.Struct.DataSize():], g)
		downstream[i] = blockStateGrad{
			BlockGrad:  blockDown[i],
			StructGrad: structGrads[i],
			DataGrad:   v[:b.Struct.DataSize()],
		}
	}

	return downstream
}

type blockRResult struct {
	Struct    RStruct
	Input     []autofunc.RResult
	InPool    []*autofunc.Variable
	BlockRes  rnn.BlockRResult
	OutStates []rnn.RState
	OutVecs   []linalg.Vector
	ROutVecs  []linalg.Vector
}

func (b *blockRResult) Outputs() []linalg.Vector {
	return b.OutVecs
}

func (b *blockRResult) ROutputs() []linalg.Vector {
	return b.ROutVecs
}

func (b *blockRResult) RStates() []rnn.RState {
	return b.OutStates
}

func (b *blockRResult) PropagateRGradient(u, uR []linalg.Vector, s []rnn.RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) []rnn.RStateGrad {
	if len(b.OutVecs) == 0 {
		return nil
	}

	if g == nil {
		g = autofunc.Gradient{}
	}

	blockUpstream := make([]linalg.Vector, len(b.OutVecs))
	blockUpstreamR := make([]linalg.Vector, len(b.OutVecs))
	blockStateUp := make([]rnn.RStateGrad, len(b.OutVecs))
	structGrads := make([]RGrad, len(b.OutVecs))
	for i, outState := range b.OutStates {
		blockUpstream[i] = make(linalg.Vector, len(b.OutVecs[i])+b.Struct.ControlSize())
		blockUpstreamR[i] = make(linalg.Vector, len(b.OutVecs[i])+b.Struct.ControlSize())
		if s != nil && s[i] != nil {
			bsg := s[i].(blockRStateGrad)
			structState := outState.(blockRState).StructState
			var ctrl, ctrlR linalg.Vector
			ctrl, ctrlR, structGrads[i] = structState.RGradient(bsg.DataGrad,
				bsg.DataGradR, bsg.StructGrad)
			copy(blockUpstream[i], ctrl)
			copy(blockUpstreamR[i], ctrlR)
			blockStateUp[i] = bsg.BlockGrad
		}
		if u != nil {
			copy(blockUpstream[i][b.Struct.ControlSize():], u[i])
			copy(blockUpstreamR[i][b.Struct.ControlSize():], uR[i])
		}
	}

	for _, v := range b.InPool {
		g[v] = make(linalg.Vector, len(v.Vector))
		rg[v] = make(linalg.Vector, len(v.Vector))
	}
	blockDown := b.BlockRes.PropagateRGradient(blockUpstream, blockUpstreamR,
		blockStateUp, rg, g)

	inputGrads := make([]linalg.Vector, len(b.OutVecs))
	inputGradsR := make([]linalg.Vector, len(b.OutVecs))
	for i, v := range b.InPool {
		inputGrads[i] = g[v]
		inputGradsR[i] = rg[v]
		delete(g, v)
		delete(rg, v)
	}

	downstream := make([]rnn.RStateGrad, len(b.OutVecs))
	for i, v := range inputGrads {
		vR := inputGradsR[i]
		b.Input[i].PropagateRGradient(v[b.Struct.DataSize():],
			vR[b.Struct.DataSize():], rg, g)
		downstream[i] = blockRStateGrad{
			BlockGrad:  blockDown[i],
			StructGrad: structGrads[i],
			DataGrad:   v[:b.Struct.DataSize()],
			DataGradR:  vR[:b.Struct.DataSize()],
		}
	}

	return downstream
}
