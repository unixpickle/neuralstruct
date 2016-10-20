package neuralstruct

import (
	"testing"

	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/rnntest"
)

func TestBlock(t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}
	b := &Block{
		Block:  rnn.NewLSTM(7, 9),
		Struct: &Stack{VectorSize: 3},
	}
	checker := rnntest.NewChecker4In(b, b)
	checker.FullCheck(t)
}
