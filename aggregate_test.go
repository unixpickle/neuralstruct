package neuralstruct

import "testing"

func TestAggregateDerivatives(t *testing.T) {
	structure := &RAggregate{
		&Stack{VectorSize: 3},
		&Queue{VectorSize: 4},
	}
	testAllDerivatives(t, structure)
}
