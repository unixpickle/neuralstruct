package neuralstruct

import (
	"errors"
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn"
)

// SeqFunc is an rnn.SeqFunc which wraps an rnn.Block
// and gives the block access to a Struct.
type SeqFunc struct {
	Block  rnn.Block
	Struct Struct
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
	structure, ok2 := slice[1].(Struct)
	if !ok1 || !ok2 {
		return nil, errors.New("invalid SeqFunc slice")
	}
	return &SeqFunc{Block: block, Struct: structure}, nil
}

// BatchSeqs applies s to a list of sequences.
func (s *SeqFunc) BatchSeqs(seqs [][]autofunc.Result) rnn.ResultSeqs {
	// TODO: this.
	return nil
}

// BatchSeqsR applies s to a list of sequences.
func (s *SeqFunc) BatchSeqsR(rv autofunc.RVector, seqs [][]autofunc.RResult) rnn.RResultSeqs {
	// TODO: this.
	return nil
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
