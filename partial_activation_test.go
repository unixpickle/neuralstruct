package neuralstruct

import (
	"math"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/weakai/neuralnet"
)

func TestPartialActivation(t *testing.T) {
	squash := func(x float64) float64 {
		s := neuralnet.Sigmoid{}
		in := &autofunc.Variable{Vector: []float64{x}}
		return s.Apply(in).Output()[0]
	}

	inputs := []float64{-1, 10, 5, -2, 3, 1, -2, 5}
	expected := []float64{-1, 10, squash(5), squash(-2), squash(3), 1, -2, 5}

	inVar := &autofunc.Variable{Vector: inputs}
	activation := &PartialActivation{
		SquashStart: 2,
		SquashEnd:   5,
		Activation:  &neuralnet.Sigmoid{},
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
