from tensor import Tensor
import torch
import numpy as np


def binary_test(data_first, data_second, binary_func):
    a_torch = torch.tensor(data_first)
    b_torch = torch.tensor(data_second)

    a_tensor = Tensor(data_first)
    b_tensor = Tensor(data_second)

    assert np.isclose(a_torch.numpy(), a_tensor.numpy()).all()
    assert np.isclose(b_torch.numpy(), b_tensor.numpy()).all()

    result_torch = binary_func(a_torch, b_torch)
    result_tensor = binary_func(a_tensor, b_tensor)
    assert np.isclose(result_torch.numpy(), result_tensor.numpy()).all()


def binary_test_grad(data_first, data_second, binary_func, loss_fn):
    a_torch = torch.tensor(data_first, requires_grad=True)
    b_torch = torch.tensor(data_second, requires_grad=True)

    a_tensor = Tensor(data_first, requires_grad=True)
    b_tensor = Tensor(data_second, requires_grad=True)

    assert np.isclose(a_torch.detach().numpy(), a_tensor.numpy()).all()
    assert np.isclose(b_torch.detach().numpy(), b_tensor.numpy()).all()

    result_torch = binary_func(a_torch, b_torch)
    result_tensor = binary_func(a_tensor, b_tensor)
    assert np.isclose(result_torch.detach().numpy(), result_tensor.numpy()).all()

    result_torch = loss_fn(result_torch)
    result_torch.backward()
    result_tensor = loss_fn(result_tensor)
    result_tensor.backward()

    assert np.isclose(a_torch.grad.numpy(), a_tensor.grad.numpy()).all()
    assert np.isclose(b_torch.grad.numpy(), b_tensor.grad.numpy()).all()


def test_basic_binary_operations():
    for binary_func in [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / y,
        lambda x, y: x @ y,
    ]:
        for a, b in [([0.001, 2.0, 3.0, 4.5], [-1.0, 2.0, 3.0, 4.1])]:
            binary_test(a, b, binary_func)

    for binary_func in [
        lambda x, y: x**y,
    ]:
        for a, b in [
            ([0.001, 2.0, 3.0, 4.5], 1.0),
            ([[0.001, 2.0, 3.0, 4.5]], 2.0),
            ([[0.001, 2.0, 3.0, 4.5]], -1.0),
            ([[0.001, 2.0, 3.0, 4.5]], 0.5),
        ]:
            binary_test(a, b, binary_func)


def test_binary_operations_broadcasting():
    for binary_func in [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / y,
    ]:
        for a, b in [
            (
                [0.001, 2.0, 3.0, 4.5],
                [[1.0, 1.0, 20.0, 3.0], [1.0, 10.0, 20.0, 3.0], [1.0, -1.0, 20.0, 3.0]],
            ),
            ([[0.001, 2.0, 3.0, 4.5]], [[[[-1.0, 2.0, 3.0, 4.1]]]]),
        ]:
            binary_test(a, b, binary_func)


def test_basic_binary_operations_grad():
    for binary_func in [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / y,
        lambda x, y: x @ y,
    ]:
        for a, b in [([0.001, 2.0, 3.0, 4.5], [-1.0, 2.0, 3.0, 4.1])]:
            for loss_fn in [
                lambda x: (x**2 - 100).sum(),
                lambda x: (x**3 + 1).sum(),
            ]:
                binary_test_grad(a, b, binary_func, loss_fn)

    for binary_func in [
        lambda x, y: x**y,
    ]:
        for a, b in [
            ([0.001, 2.0, 3.0, 4.5], 1.0),
            ([[0.001, 2.0, 3.0, 4.5]], 2.0),
            ([[0.001, 2.0, 3.0, 4.5]], -1.0),
            ([[0.001, 2.0, 3.0, 4.5]], 0.5),
        ]:
            for loss_fn in [
                lambda x: (x**2 - 100).sum(),
                lambda x: (x**3 + 1).sum(),
            ]:
                binary_test_grad(a, b, binary_func, loss_fn)


def test_binary_operations_grad_broadcasting():
    for binary_func in [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / y,
    ]:
        for a, b in [
            (
                [0.001, 2.0, 3.0, 4.5],
                [[1.0, 1.0, 20.0, 3.0], [1.0, 10.0, 20.0, 3.0], [1.0, -1.0, 20.0, 3.0]],
            ),
            ([[0.001, 2.0, 3.0, 4.5]], [[[[-1.0, 2.0, 3.0, 4.1]]]]),
        ]:
            for loss_fn in [
                lambda x: (x**2 - 100).sum(),
                lambda x: (x**3 + 1).sum(),
            ]:
                binary_test_grad(a, b, binary_func, loss_fn)
