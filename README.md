### Tensor

A simple tensor implementation similar to pytorch but with a different API.

This is part of educational pytorch-like library implementation. In the future, there will be links to other parts here, or this repository will be replaced by another, more general one.

#### Getting started:

0. Install python3

1. Install requirements:
`pip3 install -r requirements.txt` or `pip3 install -r requirements_tests.txt` (with libs for testing)

2. Run tests:
`python3 -m pytest tests/*`

3. Run examples:
`python3 examples.py`

#### Notes

At the moment, all operations are implemented quite generally. For example, operations with a gradient occur as with a tensor, that is, if desired, you can make a backward from a tensor, resulting from counting the operations of another backward.

In backward, grad from backprop is added to the current grad. So the new grad (which we will keep) will not be a leaf of the operations tree. This behavior can be changed by performing the operation inplace.
