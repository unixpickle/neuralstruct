# Overview

Originally, I had the idea to simplify the [Neural Turing Machine](https://arxiv.org/abs/1410.5401) by attaching a stack to a neural network. This would allow the network to, among other things, model [Context-free Grammars](https://en.wikipedia.org/wiki/Context-free_grammar). Once I started working on this, I realized it had [already been done](http://papers.nips.cc/paper/5648-learning-to-transduce-with-unbounded-memory.pdf). However, I still wanted my own implementation for the purposes of experimentation.

In the end, I created a more general architecture, making it theoretically possible to attach any differentiable data structure to a neural net. Currently, I have implemented a stack ([stack.go](stack.go)) and a queue ([queue.go](queue.go)). It is also possible to create aggregate structures composed of many simpler structures ([aggregate.go](aggregate.go)).
