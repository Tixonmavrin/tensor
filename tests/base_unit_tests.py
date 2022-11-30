from tensor import Tensor
import torch
import numpy as np


def test_data():
    a_torch = torch.tensor([1, 2, 3])
    a_tensor = Tensor([1, 2, 3])

    assert (a_torch.numpy() == a_tensor.numpy()).all()


def test_data_float():
    a_torch = torch.tensor([1.0, 2.0, 3.0, 4.5])
    a_tensor = Tensor([1.0, 2.0, 3.0, 4.5])

    assert (a_torch.numpy() == a_tensor.numpy()).all()


def test_data_float():
    a_tensor = Tensor([1.0, 2.0, 3.0, 4.5], requires_grad=True)

    assert (a_tensor.numpy() == a_tensor.detach().numpy()).all()
    assert not a_tensor.detach().requires_grad
