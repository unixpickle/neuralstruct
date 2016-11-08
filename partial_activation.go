package neuralstruct

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var p PartialActivation
	serializer.RegisterTypedDeserializer(p.SerializerType(), DeserializePartialActivation)
}

// An Activator suggests an activation function.
//
// Generally, a Struct will suggest an activation which
// should be applied to the raw control signal from a
// controller before it is fed to the structure.
type Activator interface {
	SuggestedActivation() neuralnet.Layer
}

// ComponentRange specifies a range of components inside
// a vector.
type ComponentRange struct {
	// Start is the (inclusive) start index.
	Start int

	// End is the (exclusive) end index.
	End int
}

// PartialActivation is an activation function which
// "squashes" sub-ranges of its inputs with an activation
// function while leaving the rest untouched.
//
// This is useful for squashing the data part of a control
// signal in a SeqFunc.
type PartialActivation struct {
	// Ranges is the list of ranges to squash, in ascending
	// order of start index.
	// This must be sorted; otherwise, the behavior of a
	// PartialActivation is undefined.
	Ranges []ComponentRange `json:"ranges"`

	// Activations are the activation functions, one per
	// each range in Ranges.
	Activations []neuralnet.Layer `json:"-"`
}

// DeserializePartialActivation deserializes a PartialActivation.
func DeserializePartialActivation(d []byte) (*PartialActivation, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) == 0 {
		return nil, errors.New("invalid PartialActivation slice")
	}
	jsonData, ok := slice[0].(serializer.Bytes)
	if !ok {
		return nil, errors.New("invalid PartialActivation slice (entry 0)")
	}
	var res PartialActivation
	if err := json.Unmarshal(jsonData, &res); err != nil {
		return nil, fmt.Errorf("invalid PartialActivation JSON: %s", err)
	}
	for i, activationObj := range slice[1:] {
		activation, ok := activationObj.(neuralnet.Layer)
		if !ok {
			return nil, fmt.Errorf("invalid PartialActivation slice (entry %d)", i+1)
		}
		res.Activations = append(res.Activations, activation)
	}
	return &res, nil
}

// Apply applies the activation function to an input.
func (p *PartialActivation) Apply(rawIn autofunc.Result) autofunc.Result {
	if len(p.Ranges) == 0 {
		return rawIn
	}
	if len(p.Ranges) != len(p.Activations) {
		panic("mismatching number of ranges and activations")
	}
	if len(rawIn.Output()) < p.Ranges[len(p.Ranges)-1].End {
		panic("input vector is too short")
	}
	return autofunc.Pool(rawIn, func(in autofunc.Result) autofunc.Result {
		var slices []autofunc.Result
		var lastCovered int
		for i, r := range p.Ranges {
			if r.Start > lastCovered {
				slices = append(slices, autofunc.Slice(in, lastCovered, r.Start))
			}
			squashMe := autofunc.Slice(in, r.Start, r.End)
			slices = append(slices, p.Activations[i].Apply(squashMe))
			lastCovered = r.End
		}
		if lastCovered < len(in.Output()) {
			slices = append(slices, autofunc.Slice(in, lastCovered, len(in.Output())))
		}
		return autofunc.Concat(slices...)
	})
}

// ApplyR is like Apply but with r-operator support.
func (p *PartialActivation) ApplyR(rv autofunc.RVector, rawIn autofunc.RResult) autofunc.RResult {
	if len(p.Ranges) == 0 {
		return rawIn
	}
	if len(p.Ranges) != len(p.Activations) {
		panic("mismatching number of ranges and activations")
	}
	if len(rawIn.Output()) < p.Ranges[len(p.Ranges)-1].End {
		panic("input vector is too short")
	}
	return autofunc.PoolR(rawIn, func(in autofunc.RResult) autofunc.RResult {
		var slices []autofunc.RResult
		var lastCovered int
		for i, r := range p.Ranges {
			if r.Start > lastCovered {
				slices = append(slices, autofunc.SliceR(in, lastCovered, r.Start))
			}
			squashMe := autofunc.SliceR(in, r.Start, r.End)
			slices = append(slices, p.Activations[i].ApplyR(rv, squashMe))
			lastCovered = r.End
		}
		if lastCovered < len(in.Output()) {
			slices = append(slices, autofunc.SliceR(in, lastCovered, len(in.Output())))
		}
		return autofunc.ConcatR(slices...)
	})
}

// SerializerType returns the unique ID used to serialize
// PartialActivations with the serializer package.
func (p *PartialActivation) SerializerType() string {
	return "github.com/unixpickle/neuralstruct.PartialActivation"
}

// Serialize serializes the PartialActivation.
func (p *PartialActivation) Serialize() ([]byte, error) {
	jsonData, err := json.Marshal(p)
	if err != nil {
		return nil, err
	}
	slice := []serializer.Serializer{serializer.Bytes(jsonData)}
	for _, act := range p.Activations {
		slice = append(slice, act)
	}
	return serializer.SerializeSlice(slice)
}
