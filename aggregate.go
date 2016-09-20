package neuralstruct

import (
	"fmt"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

func init() {
	var a Aggregate
	var r RAggregate
	serializer.RegisterTypedDeserializer(a.SerializerType(), DeserializeAggregate)
	serializer.RegisterTypedDeserializer(r.SerializerType(), DeserializeRAggregate)
}

// Aggregate combines multiple Structs into a single,
// compound struct.
type Aggregate []Struct

// DeserializeAggregate deserializes an Aggregate.
func DeserializeAggregate(d []byte) (Aggregate, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	res := make(Aggregate, len(slice))
	for i, x := range slice {
		if s, ok := x.(Struct); !ok {
			return nil, fmt.Errorf("deserialized type is not Struct: %T", x)
		} else {
			res[i] = s
		}
	}
	return res, nil
}

// ControlSize returns the sum of the control sizes
// of every structure in the aggregate.
func (a Aggregate) ControlSize() int {
	var sum int
	for _, s := range a {
		sum += s.ControlSize()
	}
	return sum
}

// DataSize returns the sum of the data sizes of every
// structure in the aggregate.
func (a Aggregate) DataSize() int {
	var sum int
	for _, s := range a {
		sum += s.DataSize()
	}
	return sum
}

// StartState returns a state representing the aggregate
// start state.
// The data and control vectors of this state and its
// descendants are found by concatenating those of the
// aggregated sub-states.
func (a Aggregate) StartState() State {
	var res aggregateState
	for _, s := range a {
		state := s.StartState()
		res.Structs = append(res.Structs, s)
		res.States = append(res.States, state)
		res.JoinedData = append(res.JoinedData, state.Data()...)
	}
	return &res
}

// SerializerType returns the unique ID used to serialize
// Aggregates with the serializer package.
func (a *Aggregate) SerializerType() string {
	return "github.com/unixpickle/neuralstruct.Aggregate"
}

// Serialize encodes the aggregate, given that all of the
// contained structs are serializers.
func (a Aggregate) Serialize() ([]byte, error) {
	var serializers []serializer.Serializer
	for _, x := range a {
		if s, ok := x.(serializer.Serializer); ok {
			serializers = append(serializers, s)
		} else {
			return nil, fmt.Errorf("struct is not a serializer: %T", x)
		}
	}
	return serializer.SerializeSlice(serializers)
}

// An RAggregate is like an Aggregate, but with support
// for the r-operator.
type RAggregate []RStruct

// DeserializeRAggregate deserializes an RAggregate.
func DeserializeRAggregate(d []byte) (RAggregate, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	res := make(RAggregate, len(slice))
	for i, x := range slice {
		if r, ok := x.(RStruct); !ok {
			return nil, fmt.Errorf("deserialized type is not RStruct: %T", x)
		} else {
			res[i] = r
		}
	}
	return res, nil
}

// ControlSize is like Aggregate.ControlSize().
func (r RAggregate) ControlSize() int {
	return r.aggregate().ControlSize()
}

// DataSize is like Aggregate.DataSize().
func (r RAggregate) DataSize() int {
	return r.aggregate().DataSize()
}

// StartState is like Aggregate.StartState().
func (r RAggregate) StartState() State {
	return r.aggregate().StartState()
}

// StartRState is like StartState, but for RStates.
func (r RAggregate) StartRState() RState {
	var res aggregateRState
	for _, s := range r {
		state := s.StartRState()
		res.Structs = append(res.Structs, s)
		res.States = append(res.States, state)
		res.JoinedData = append(res.JoinedData, state.Data()...)
		res.JoinedRData = append(res.JoinedRData, state.RData()...)
	}
	return &res
}

// SerializerType returns the unique ID used to serialize
// RAggregates with the serializer package.
func (r RAggregate) SerializerType() string {
	return "github.com/unixpickle/neuralstruct.RAggregate"
}

// Serialize encodes the aggregate, given that all of the
// contained structs are serializers.
func (r RAggregate) Serialize() ([]byte, error) {
	var serializers []serializer.Serializer
	for _, x := range r {
		if s, ok := x.(serializer.Serializer); ok {
			serializers = append(serializers, s)
		} else {
			return nil, fmt.Errorf("struct is not a serializer: %T", x)
		}
	}
	return serializer.SerializeSlice(serializers)
}

func (r RAggregate) aggregate() Aggregate {
	a := make(Aggregate, len(r))
	for i, s := range r {
		a[i] = s
	}
	return a
}

type aggregateState struct {
	Structs    []Struct
	States     []State
	JoinedData linalg.Vector
}

func (a *aggregateState) Data() linalg.Vector {
	return a.JoinedData
}

func (a *aggregateState) Gradient(upstream linalg.Vector, grad Grad) (linalg.Vector, Grad) {
	var gradList []Grad
	if grad != nil {
		gradList = grad.([]Grad)
	}
	var dataIdx int
	var downstream linalg.Vector
	var downstreamGrad []Grad
	for i, s := range a.States {
		dataSize := a.Structs[i].DataSize()
		subUpstream := upstream[dataIdx : dataIdx+dataSize]
		dataIdx += dataSize

		var subDownstream linalg.Vector
		var subGrad Grad
		if gradList == nil {
			subDownstream, subGrad = s.Gradient(subUpstream, nil)
		} else {
			subDownstream, subGrad = s.Gradient(subUpstream, gradList[i])
		}
		downstream = append(downstream, subDownstream...)
		downstreamGrad = append(downstreamGrad, subGrad)
	}
	return downstream, downstreamGrad
}

func (a *aggregateState) NextState(ctrl linalg.Vector) State {
	newState := &aggregateState{Structs: a.Structs}
	var ctrlIdx int
	for i, s := range a.States {
		ctrlSize := a.Structs[i].ControlSize()
		subCtrl := ctrl[ctrlIdx : ctrlIdx+ctrlSize]
		ctrlIdx += ctrlSize
		newSub := s.NextState(subCtrl)
		newState.States = append(newState.States, newSub)
		newState.JoinedData = append(newState.JoinedData, newSub.Data()...)
	}
	return newState
}

type aggregateRState struct {
	Structs     []RStruct
	States      []RState
	JoinedData  linalg.Vector
	JoinedRData linalg.Vector
}

func (a *aggregateRState) Data() linalg.Vector {
	return a.JoinedData
}

func (a *aggregateRState) RData() linalg.Vector {
	return a.JoinedRData
}

func (a *aggregateRState) RGradient(upstream, upstreamR linalg.Vector,
	grad RGrad) (linalg.Vector, linalg.Vector, RGrad) {
	var gradList []RGrad
	if grad != nil {
		gradList = grad.([]RGrad)
	}
	var dataIdx int
	var downstream linalg.Vector
	var downstreamR linalg.Vector
	var downstreamGrad []RGrad
	for i, s := range a.States {
		dataSize := a.Structs[i].DataSize()
		subUpstream := upstream[dataIdx : dataIdx+dataSize]
		subUpstreamR := upstreamR[dataIdx : dataIdx+dataSize]
		dataIdx += dataSize

		var subDownstream linalg.Vector
		var subDownstreamR linalg.Vector
		var subGrad RGrad
		if gradList == nil {
			subDownstream, subDownstreamR, subGrad = s.RGradient(subUpstream,
				subUpstreamR, nil)
		} else {
			subDownstream, subDownstreamR, subGrad = s.RGradient(subUpstream,
				subUpstreamR, gradList[i])
		}
		downstream = append(downstream, subDownstream...)
		downstreamR = append(downstreamR, subDownstreamR...)
		downstreamGrad = append(downstreamGrad, subGrad)
	}
	return downstream, downstreamR, downstreamGrad
}

func (a *aggregateRState) NextRState(ctrl, ctrlR linalg.Vector) RState {
	newState := &aggregateRState{Structs: a.Structs}
	var ctrlIdx int
	for i, s := range a.States {
		ctrlSize := a.Structs[i].ControlSize()
		subCtrl := ctrl[ctrlIdx : ctrlIdx+ctrlSize]
		subCtrlR := ctrlR[ctrlIdx : ctrlIdx+ctrlSize]
		ctrlIdx += ctrlSize
		newSub := s.NextRState(subCtrl, subCtrlR)
		newState.States = append(newState.States, newSub)
		newState.JoinedData = append(newState.JoinedData, newSub.Data()...)
		newState.JoinedRData = append(newState.JoinedRData, newSub.RData()...)
	}
	return newState
}
