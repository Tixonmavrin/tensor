A simple tensor implementation similar to pytorch but with a different API.

0. Install python3

1. Install requirements:
`pip3 install -r requirements.txt` or `pip3 install -r requirements_tests.txt` (with libs for testing)

2. Run tests:
`python3 -m pytest tests/*`

3. Run examples:
`python3 examples.py`

Currently, in the backward function, the old tensor gradient is replaced by the new one. 

It is also possible to make a backprop for the gradient that we pass to the backward function. 

This behavior is worth revisiting. The gradients may need to be copied beforehand.