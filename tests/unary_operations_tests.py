from tensor import Tensor
import torch
import numpy as np


def unary_test(data, unary_func):
    a_torch = torch.tensor(data)
    a_tensor = Tensor(data)
    assert np.isclose(a_torch.numpy(), a_tensor.numpy()).all()

    result_torch = unary_func(a_torch)
    result_tensor = unary_func(a_tensor)
    assert np.isclose(result_torch.detach().numpy(), result_tensor.numpy()).all()


def unary_test_grad(data, unary_func, loss_fn):
    a_torch = torch.tensor(data, requires_grad=True, dtype=torch.float64)
    a_tensor = Tensor(data, requires_grad=True, dtype=np.float64)
    assert np.isclose(a_torch.detach().numpy(), a_tensor.numpy()).all()

    result_torch = unary_func(a_torch)
    result_tensor = unary_func(a_tensor)
    assert np.isclose(result_torch.detach().numpy(), result_tensor.numpy()).all()

    result_torch = loss_fn(result_torch)
    result_torch.backward()
    result_tensor = loss_fn(result_tensor)
    result_tensor.backward()
    assert np.isclose(a_torch.grad.numpy(), a_tensor.grad.numpy()).all()


def test_unary_operations_positive():
    for unary_func in [
        lambda x: x.exp(),
        lambda x: -x,
        lambda x: x.sum(),
        lambda x: x.sum(0),
        lambda x: x.sum(-1),
        lambda x: x.sum([0]),
        lambda x: x.sum([-1]),
        lambda x: x.mean(),
        lambda x: x.mean(0),
        lambda x: x.mean(-1),
        lambda x: x.mean([0]),
        lambda x: x.mean([-1]),
        lambda x: x.relu(),
        lambda x: x.leaky_relu()
        if isinstance(x, Tensor)
        else torch.nn.functional.leaky_relu(x),
        lambda x: x.softmax(0),
        lambda x: x.softmax(-1),
        lambda x: x.log_softmax(0),
        lambda x: x.log_softmax(-1),
        lambda x: x.sigmoid(),
        lambda x: x.dropout(training=False)
        if isinstance(x, Tensor)
        else torch.nn.functional.dropout(x, training=False),
    ]:
        for data in [
            [0.001, 2.0, 3.0, 4.5],
            [1.0, 2.0, 3.0, 4.0],
            [
                [
                    [0.001, 2.0, 3.0, 4.5],
                    [1.0, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.55],
                ]
            ],
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        ]:
            unary_test(data, unary_func)


def test_unary_operations_grad_positive():
    for unary_func in [
        lambda x: x.exp(),
        lambda x: -x,
        lambda x: x.sum(),
        lambda x: x.sum(0),
        lambda x: x.sum(-1),
        lambda x: x.sum([0]),
        lambda x: x.sum([-1]),
        lambda x: x.mean(),
        lambda x: x.mean(0),
        lambda x: x.mean(-1),
        lambda x: x.mean([0]),
        lambda x: x.mean([-1]),
        lambda x: x.relu(),
        lambda x: x.leaky_relu()
        if isinstance(x, Tensor)
        else torch.nn.functional.leaky_relu(x),
        lambda x: x.softmax(0),
        lambda x: x.softmax(-1),
        lambda x: x.log_softmax(0),
        lambda x: x.log_softmax(-1),
        lambda x: x.sigmoid(),
        lambda x: x.dropout(training=False)
        if isinstance(x, Tensor)
        else torch.nn.functional.dropout(x, training=False),
    ]:
        for data in [
            [0.001, 2.0, 3.0, 4.5],
            [1.0, 2.0, 3.0, 4.0],
            [
                [
                    [0.001, 2.0, 3.0, 4.5],
                    [1.0, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.55],
                ]
            ],
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        ]:
            for loss_fn in [
                lambda x: (x**2 - 100).sum(),
                lambda x: (x**3 + 1).sum(),
            ]:
                unary_test_grad(data, unary_func, loss_fn)


def test_unary_operations_with_negative():
    for unary_func in [
        lambda x: x.exp(),
        lambda x: -x,
        lambda x: x.sum(),
        lambda x: x.sum(0),
        lambda x: x.sum(-1),
        lambda x: x.sum([0]),
        lambda x: x.sum([-1]),
        lambda x: x.mean(),
        lambda x: x.mean(0),
        lambda x: x.mean(-1),
        lambda x: x.mean([0]),
        lambda x: x.mean([-1]),
        lambda x: x.relu(),
        lambda x: x.leaky_relu()
        if isinstance(x, Tensor)
        else torch.nn.functional.leaky_relu(x),
        lambda x: x.softmax(0),
        lambda x: x.softmax(-1),
        lambda x: x.log_softmax(0),
        lambda x: x.log_softmax(-1),
        lambda x: x.sigmoid(),
        lambda x: x.dropout(training=False)
        if isinstance(x, Tensor)
        else torch.nn.functional.dropout(x, training=False),
    ]:
        for data in [
            [-0.001, -2.0, -3.0, -4.5],
            [-1.0, -2.0, -3.0, -4.0],
            [0.0, -2.0, 4.0, 0.0001],
            [
                [
                    [0.001, 2.0, 3.0, -4.5],
                    [-1.0, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.55],
                ]
            ],
            [[1.0, 2.0, -3.0, 4.0], [-1.0, -2.0, 3.0, -4.0]],
        ]:
            unary_test(data, unary_func)


def test_unary_operations_grad_with_negative():
    for unary_func in [
        lambda x: x.exp(),
        lambda x: -x,
        lambda x: x.sum(),
        lambda x: x.sum(0),
        lambda x: x.sum(-1),
        lambda x: x.sum([0]),
        lambda x: x.sum([-1]),
        lambda x: x.mean(),
        lambda x: x.mean(0),
        lambda x: x.mean(-1),
        lambda x: x.mean([0]),
        lambda x: x.mean([-1]),
        lambda x: x.relu(),
        lambda x: x.leaky_relu()
        if isinstance(x, Tensor)
        else torch.nn.functional.leaky_relu(x),
        lambda x: x.softmax(0),
        lambda x: x.softmax(-1),
        lambda x: x.log_softmax(0),
        lambda x: x.log_softmax(-1),
        lambda x: x.sigmoid(),
        lambda x: x.dropout(training=False)
        if isinstance(x, Tensor)
        else torch.nn.functional.dropout(x, training=False),
    ]:
        for data in [
            [-0.001, -2.0, -3.0, -4.5],
            [-1.0, -2.0, -3.0, -4.0],
            [0.0, -2.0, 4.0, 0.0001],
            [
                [
                    [0.001, 2.0, 3.0, -4.5],
                    [-1.0, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.55],
                ]
            ],
            [[1.0, 2.0, -3.0, 4.0], [-1.0, -2.0, 3.0, -4.0]],
        ]:
            for loss_fn in [
                lambda x: (x**2 - 100).sum(),
                lambda x: (x**3 + 1).sum(),
            ]:
                unary_test_grad(data, unary_func, loss_fn)


def test_unary_format_operations():
    for unary_func in [
        lambda x: x.unsqueeze(0),
        lambda x: x.unsqueeze(1),
        lambda x: x.transpose(0, 1),
        lambda x: x.transpose(1, 0),
        lambda x: x.permute(1, 0, 2),
        lambda x: x.permute(2, 0, 1),
        lambda x: x.T,
        lambda x: x.view(2, 8),
        lambda x: x.view(1, 1, -1),
        lambda x: x.squeeze(0),
    ]:
        for data in [
            [
                [
                    [0.001, 2.0, 3.0, -4.5],
                    [-1.0, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.55],
                ]
            ]
        ]:
            unary_test(data, unary_func)


def test_unary_format_operations_grad():
    for unary_func in [
        lambda x: x.unsqueeze(0),
        lambda x: x.unsqueeze(1),
        lambda x: x.transpose(0, 1),
        lambda x: x.transpose(1, 0),
        lambda x: x.permute(1, 0, 2),
        lambda x: x.permute(2, 0, 1),
        lambda x: x.T,
        lambda x: x.view(2, 8),
        lambda x: x.view(1, 1, -1),
        lambda x: x.squeeze(0),
    ]:
        for data in [
            [
                [
                    [0.001, 2.0, 3.0, -4.5],
                    [-1.0, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.5],
                    [0.1, 2.0, 3.0, 4.55],
                ]
            ]
        ]:
            for loss_fn in [
                lambda x: (x**2 - 100).sum(),
                lambda x: (x**3 + 1).sum(),
            ]:
                unary_test_grad(data, unary_func, loss_fn)
