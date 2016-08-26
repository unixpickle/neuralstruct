package neuralstruct

import (
	"math"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func TestPartialActivation(t *testing.T) {
	squash := func(x float64) float64 {
		s := neuralnet.Sigmoid{}
		in := &autofunc.Variable{Vector: []float64{x}}
		return s.Apply(in).Output()[0]
	}

	inputs := []float64{-1, 10, 5, -2, 3, 1, -2, 5}
	expected := []float64{-1, 10, squash(5), squash(-2), squash(3), 1, -2, math.Tanh(5)}

	inVar := &autofunc.Variable{Vector: inputs}
	activation := &PartialActivation{
		Ranges: []ComponentRange{
			{2, 5},
			{7, 8},
		},
		Activations: []neuralnet.Layer{
			&neuralnet.Sigmoid{},
			&neuralnet.HyperbolicTangent{},
		},
	}
	actual := activation.Apply(inVar).Output()

	if len(actual) != len(expected) {
		t.Fatalf("expected len %d got len %d", len(expected), len(actual))
	}

	for i, x := range expected {
		a := actual[i]
		if math.Abs(x-a) > 1e-5 {
			t.Errorf("entry %d should be %f but it's %f", i, x, a)
		}
	}
}

func TestPartialActivationSerialize(t *testing.T) {
	activation := &PartialActivation{
		Ranges: []ComponentRange{
			{2, 5},
			{7, 8},
		},
		Activations: []neuralnet.Layer{
			&neuralnet.Sigmoid{},
			&neuralnet.HyperbolicTangent{},
		},
	}
	data, err := serializer.SerializeWithType(activation)
	if err != nil {
		t.Fatal(err)
	}
	newActivation, err := serializer.DeserializeWithType(data)
	if err != nil {
		t.Fatal(err)
	}
	act, ok := newActivation.(*PartialActivation)
	if !ok {
		t.Fatalf("type shouldn't be %T", newActivation)
	}
	if len(act.Ranges) != 2 || len(act.Activations) != 2 {
		t.Fatalf("lens are %d,%d not 2,2", len(act.Ranges), len(act.Activations))
	}

	_, ok1 := act.Activations[0].(*neuralnet.Sigmoid)
	_, ok2 := act.Activations[1].(*neuralnet.HyperbolicTangent)
	if !ok1 || !ok2 {
		t.Fatalf("types shouldn't be %T,%T", act.Activations[0], act.Activations[1])
	}
}
