package neuralstruct

import (
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var p PartialActivation
	serializer.RegisterTypedDeserializer(p.SerializerType(), DeserializePartialActivation)
}

// PartialActivation is an activation function which
// squashes a sub-range of its inputs using an activation
// function while leaving the rest untouched.
//
// This is useful for squashing the data part of a control
// signal in a SeqFunc.
type PartialActivation struct {
	// SquashStart is the start index (inclusive) of the
	// activation part of input vectors.
	SquashStart int

	// SquashEnd is the end index (exclusive) of the
	// activation part of input vectors.
	SquashEnd int

	// Activation is the activation function to partially
	// apply to input vectors.
	Activation neuralnet.Layer
}

// DeserializePartialActivation deserializes a PartialActivation.
func DeserializePartialActivation(d []byte) (*PartialActivation, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 3 {
		return nil, errors.New("invalid PartialActivation slice")
	}
	start, ok1 := slice[0].(serializer.Int)
	end, ok2 := slice[1].(serializer.Int)
	activation, ok3 := slice[2].(neuralnet.Layer)
	if !ok1 || !ok2 || !ok3 {
		return nil, errors.New("invalid PartialActivation slice")
	}
	return &PartialActivation{
		SquashStart: int(start),
		SquashEnd:   int(end),
		Activation:  activation,
	}, nil
}

// Apply applies the activation function to an input.
func (p *PartialActivation) Apply(rawIn autofunc.Result) autofunc.Result {
	if len(rawIn.Output()) < p.SquashEnd {
		panic("input vector is too short")
	}
	return autofunc.Pool(rawIn, func(in autofunc.Result) autofunc.Result {
		preSquash := autofunc.Slice(in, 0, p.SquashStart)
		squash := autofunc.Slice(in, p.SquashStart, p.SquashEnd)
		postSquash := autofunc.Slice(in, p.SquashEnd, len(in.Output()))
		squash = p.Activation.Apply(squash)
		return autofunc.Concat(preSquash, squash, postSquash)
	})
}

// ApplyR is like Apply but with r-operator support.
func (p *PartialActivation) ApplyR(rv autofunc.RVector, rawIn autofunc.RResult) autofunc.RResult {
	if len(rawIn.Output()) < p.SquashEnd {
		panic("input vector is too short")
	}
	return autofunc.PoolR(rawIn, func(in autofunc.RResult) autofunc.RResult {
		preSquash := autofunc.SliceR(in, 0, p.SquashStart)
		squash := autofunc.SliceR(in, p.SquashStart, p.SquashEnd)
		postSquash := autofunc.SliceR(in, p.SquashEnd, len(in.Output()))
		squash = p.Activation.ApplyR(rv, squash)
		return autofunc.ConcatR(preSquash, squash, postSquash)
	})
}

// SerializerType returns the unique ID used to serialize
// PartialActivations with the serializer package.
func (p *PartialActivation) SerializerType() string {
	return "github.com/unixpickle/neuralstruct.PartialActivation"
}

// Serialize serializes the PartialActivation.
func (p *PartialActivation) Serialize() ([]byte, error) {
	slice := []serializer.Serializer{
		serializer.Int(p.SquashStart),
		serializer.Int(p.SquashEnd),
		p.Activation,
	}
	return serializer.SerializeSlice(slice)
}
