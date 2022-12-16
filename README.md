## Tensor

A simple tensor implementation similar to pytorch but with a different API.

This is part of educational pytorch-like library implementation. In the future, there will be links to other parts here.

### Getting started (1 option):

0. Install python3, git and do `https://github.com/Tixonmavrin/tensor`

1. Install requirements:
`pip3 install -r requirements.txt` or `pip3 install -r requirements_tests.txt` (with libs for testing)

2. Run tests:
`python3 -m pytest tests/*` (if you installed the libraries for testing)

3. Run examples:
`python3 examples.py`

### Notes

At the moment, all operations are implemented quite generally. For example, operations with a gradient occur as with a tensor, that is, if desired, you can make a backward from a tensor, resulting from counting the operations of another backward.

In backward, grad from backprop is added to the current grad. So the new grad (which we will keep) will not be a leaf of the operations tree. This behavior can be changed in future.

### Features

- [x] Cache of grad
- [x] Grad operations in backward

#### Operations
- [x] Add
- [x] Substract
- [x] Muliply
- [x] Divide
- [x] Power (number)
- [x] Power (tensor) (may be add eps in future)
- [x] Power (tensor, in even degrees)
- [x] Matmul
- [x] Negative
- [x] Exponential
- [x] Natural Logarithm
- [x] Argmax (no grad)
- [x] View
- [x] Squeeze
- [x] Transpose
- [x] Permute
- [x] .T
- [x] Unsqueeze
- [x] Sum
- [x] Mean
- [x] Softmax
- [x] Log Softmax
- [x] Dropout
- [ ] Max
- [ ] Min
- [ ] Median
- [ ] Std
- [ ] Get item
- [ ] Padding
- [ ] Batch Normalization

#### Activation Functions

- [x] ReLU
- [x] Leaky ReLU
- [x] Sigmoid
- [ ] Elu
- [ ] Tanh
- [ ] GELU
- [ ] SoftPlus

#### Loss Functions

- [ ] Cross Entropy
- [ ] Negative Log Likelihood
- [ ] Mean Squared Error
- [ ] Binary Cross Entropy

#### Initializers

- [x] Fill with zeros / ones / other given constants
- [x] Uniform / Normal
- [ ] Xavier (Glorot) uniform / normal ([Understanding the Difficulty of Training Deep Feedforward Neural Networks.](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) *Xavier Glorot and Yoshua Bengio.* AISTATS 2010.)
- [ ] Kaiming (He) uniform / normal ([Delving Deep into Rectifiers: Surpassing Human-level Performance on ImageNet Classification.](https://arxiv.org/pdf/1502.01852.pdf) *Kaiming He, et al.* ICCV 2015.)
- [ ] LeCun uniform / normal ([Efficient Backprop.](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) *Yann LeCun, et al.* 1998.)

#### Other operations and layers

- [ ] Linear
- [ ] Convolution (1D / 2D)
- [ ] MaxPooling (1D / 2D)
- [ ] Unfold
- [ ] RNN
- [ ] Flatten
- [ ] Sequential

#### Optimizers (apply to current tensor)

- [ ] SGD
- [ ] Momentum
- [ ] Adagrad
- [ ] RMSprop
- [ ] Adadelta
- [ ] Adam
