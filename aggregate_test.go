package neuralstruct

import (
	"reflect"
	"testing"

	"github.com/unixpickle/weakai/neuralnet"
)

func TestAggregateDerivatives(t *testing.T) {
	structure := RAggregate{
		&Stack{VectorSize: 3},
		&Queue{VectorSize: 4},
	}
	testAllDerivatives(t, structure)
}

func TestAggregateActivation(t *testing.T) {
	structure := RAggregate{
		&Stack{VectorSize: 3},
		&Queue{VectorSize: 7},
		&Stack{VectorSize: 2, NoReplace: true},
		RAggregate{
			&Stack{VectorSize: 1},
			&Queue{VectorSize: 4},
		},
	}
	actual := structure.SuggestedActivation()
	expected := &PartialActivation{
		Ranges: []ComponentRange{
			{Start: 0, End: 7},
			{Start: 7, End: 17},
			{Start: 17, End: 22},
			{Start: 22, End: 34},
		},
		Activations: []neuralnet.Layer{
			&PartialActivation{
				Ranges:      []ComponentRange{{Start: 4, End: 7}},
				Activations: []neuralnet.Layer{&neuralnet.HyperbolicTangent{}},
			},
			&PartialActivation{
				Ranges:      []ComponentRange{{Start: 3, End: 10}},
				Activations: []neuralnet.Layer{&neuralnet.HyperbolicTangent{}},
			},
			&PartialActivation{
				Ranges:      []ComponentRange{{Start: 3, End: 5}},
				Activations: []neuralnet.Layer{&neuralnet.HyperbolicTangent{}},
			},
			&PartialActivation{
				Ranges: []ComponentRange{
					{Start: 0, End: 5},
					{Start: 5, End: 12},
				},
				Activations: []neuralnet.Layer{
					&PartialActivation{
						Ranges:      []ComponentRange{{Start: 4, End: 5}},
						Activations: []neuralnet.Layer{&neuralnet.HyperbolicTangent{}},
					},
					&PartialActivation{
						Ranges:      []ComponentRange{{Start: 3, End: 7}},
						Activations: []neuralnet.Layer{&neuralnet.HyperbolicTangent{}},
					},
				},
			},
		},
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("unexpected activation: %v expected %v", actual, expected)
	}
}
