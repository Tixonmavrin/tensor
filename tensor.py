import numpy as np
from typing import Union, Tuple, Optional, Callable

ArrayLike = Union[float, list, np.ndarray]


def to_ndarray(data: ArrayLike) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data)


def is_tensor(data: "Tensor"):
    return isinstance(data, Tensor)


def to_tensor(data: "Tensor"):
    return data if isinstance(data, Tensor) else Tensor(data)


def reshape_op_binary(data: "Tensor", true_shape: Tuple[int]):
    diff = data.ndim - len(true_shape)
    unmatched_axis = tuple(
        i + diff for i, s in enumerate(true_shape) if s != data.shape[i + diff]
    )
    unmatched_unsqueezed = tuple(range(diff))

    data = data.sum(dim=unmatched_axis, keepdims=True).sum(
        dim=unmatched_unsqueezed, keepdims=False
    )
    if data.shape != true_shape:
        raise RuntimeError("Bad reshape. If you see this error, open issue, please")
    return data


class Tensor:
    EPS = 1e-12
    BASEDTYPE = np.float32

    def __init__(self, data: "Tensor", requires_grad: bool = False, dtype: str = None):
        if is_tensor(data):
            self._data = data._data
        else:
            self._data = to_ndarray(data)
        self._requires_grad = requires_grad

        self._grad = None
        self._grad_fn = None

        if dtype is not None:
            self._dtype = dtype
            self._data = self._data.astype(self._dtype)
        else:
            self._dtype = self._data.dtype

    @property
    def dtype(self) -> np.ndarray:
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: str) -> None:
        self._dtype = new_dtype
        self._data = self._data.astype(self._dtype)

    @property
    def data(self) -> np.ndarray:
        return Tensor(self._data)

    @data.setter
    def data(self, other_tensor: "Tensor") -> None:
        if is_tensor(other_tensor):
            self._data = other_tensor._data
            self._dtype = other_tensor._dtype
        else:
            self._data = to_ndarray(other_tensor)
            self._data = self._data.astype(self._dtype)

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool = False) -> None:
        self._requires_grad = requires_grad

    @property
    def grad(self) -> "Tensor":
        return self._grad

    @grad.setter
    def grad(self, grad: ArrayLike) -> None:
        grad = to_tensor(grad)
        self._grad = grad

    @property
    def grad_fn(self) -> Callable:
        return self._grad_fn

    @grad_fn.setter
    def grad_fn(self, grad_fn: np.ndarray) -> None:
        self._grad_fn = grad_fn

    def numpy(self):
        return self._data

    def detach(self) -> "Tensor":
        return Tensor(data=self._data)

    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape, dtype=cls.BASEDTYPE), **kwargs)

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape, dtype=cls.BASEDTYPE), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.randn(*shape).astype(cls.BASEDTYPE), **kwargs)

    def fill_(self, val: float) -> None:
        self._data.fill(val)

    def zero_(self) -> None:
        self.fill_(0.0)

    def one_(self) -> None:
        self.fill_(1.0)

    def uniform_(self, low: float = 0.0, high: float = 1.0) -> None:
        self._data = np.random.uniform(low=low, high=high, size=self.shape)

    def normal_(self, mean: float = 0.0, std: float = 1.0) -> None:
        self._data = np.random.normal(loc=mean, scale=std, size=self.shape)

    @property
    def shape(self) -> Tuple[int]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def numel(self) -> int:
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    def size(self, dim: Optional[int] = None) -> Union[int, Tuple[int]]:
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def dim(self) -> int:
        return self.ndim

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        if grad is None:
            grad = Tensor(1.0)

        if self.requires_grad:
            if self._grad is None:
                self._grad = Tensor.zeros(*self.shape)
            self._grad += grad

        if self._grad_fn is not None:
            self._grad_fn(grad)

    def _add(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            data=self._data + other._data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        if out.requires_grad:

            def grad_add(grad: "Tensor") -> None:
                self.backward(reshape_op_binary(grad, self._data.shape))
                other.backward(reshape_op_binary(grad, other._data.shape))

            out._grad_fn = grad_add
        return out

    def __add__(self, other: "Tensor") -> "Tensor":
        other = to_tensor(other)
        return self._add(other)

    def __radd__(self, other: "Tensor") -> "Tensor":
        other = to_tensor(other)
        return other._add(self)

    def _sub(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            data=self._data - other._data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        if out.requires_grad:

            def grad_sub(grad: "Tensor") -> None:
                self.backward(reshape_op_binary(grad, self._data.shape))
                other.backward(reshape_op_binary(-grad, other._data.shape))

            out._grad_fn = grad_sub
        return out

    def __sub__(self, other: "Tensor") -> "Tensor":
        other = to_tensor(other)
        return self._sub(other)

    def __rsub__(self, other: "Tensor") -> "Tensor":
        other = to_tensor(other)
        return other._sub(self)

    def _mul(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            data=self._data * other._data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        if out.requires_grad:

            def grad_mul(grad: "Tensor") -> None:
                self.backward(reshape_op_binary(grad * other._data, self._data.shape))
                other.backward(reshape_op_binary(grad * self._data, other._data.shape))

            out._grad_fn = grad_mul
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        other = to_tensor(other)
        return self._mul(other)

    def __rmul__(self, other: "Tensor") -> "Tensor":
        other = to_tensor(other)
        return other._mul(self)

    def _truediv(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            data=self._data / other._data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        if out.requires_grad:

            def grad_div(grad: "Tensor") -> None:
                self.backward(reshape_op_binary(grad / other._data, self._data.shape))
                other.backward(
                    reshape_op_binary(
                        -grad * self._data / (other._data**2), other._data.shape
                    )
                )

            out._grad_fn = grad_div
        return out

    def __truediv__(self, other: "Tensor") -> "Tensor":
        other = to_tensor(other)
        return self._truediv(other)

    def __rtruediv__(self, other: "Tensor") -> "Tensor":
        other = to_tensor(other)
        return other._truediv(self)

    def _powtensor(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            data=self._data**other._data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        if out.requires_grad:

            def grad_pow(grad: "Tensor") -> None:
                self.backward(
                    reshape_op_binary(
                        grad * other._data * (self._data ** (other._data - 1.0)),
                        self._data.shape,
                    )
                )
                # TODO: check that epsilon usage is correct here
                # may be just fill 0 or use only log (without epsilon)
                other.backward(
                    reshape_op_binary(
                        grad * out._data * np.log(self._data + Tensor.EPS),
                        other._data.shape,
                    )
                )

            out._grad_fn = grad_pow
        return out

    def _pow(self, other: float) -> "Tensor":
        out = Tensor(data=self._data**other, requires_grad=self.requires_grad)
        if out.requires_grad:

            def grad_pow(grad: "Tensor") -> None:
                self.backward(
                    reshape_op_binary(
                        grad * other * (self._data ** (other - 1)), self._data.shape
                    )
                )

            out._grad_fn = grad_pow
        return out

    def __pow__(self, other: "Tensor") -> "Tensor":
        if is_tensor(other):
            return self._powtensor(other)
        return self._pow(other)

    def __rpow__(self, other: "Tensor") -> "Tensor":
        other = to_tensor(other)
        return other._powtensor(self)

    def _matmul(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            data=np.dot(self._data, other._data),
            requires_grad=self.requires_grad or other.requires_grad,
        )
        if out.requires_grad:

            def grad_mm(grad: "Tensor") -> None:
                self_data = self._data
                other_data = other._data

                # TODO: remove this block
                # May be use benefits of np.dot
                if grad.ndim <= 1:
                    grad = grad.view(1, -1)
                if other_data.ndim <= 1:
                    other_data = other_data.reshape(-1, 1)
                if self_data.ndim <= 1:
                    self_data = self_data.reshape(1, -1)

                self.backward((grad @ other_data.T).view(*self._data.shape))
                # TODO: Use .T here instead of __rmatmul__
                other.backward((grad.__rmatmul__(self_data.T)).view(*other._data.shape))

            out._grad_fn = grad_mm
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = to_tensor(other)
        return self._matmul(other)

    def __rmatmul__(self, other: "Tensor") -> "Tensor":
        other = to_tensor(other)
        return other._matmul(self)

    def __neg__(self) -> "Tensor":
        out = Tensor(data=-self._data, requires_grad=self.requires_grad)
        if out.requires_grad:

            def grad_neg(grad: "Tensor") -> None:
                self.backward(-grad)

            out._grad_fn = grad_neg
        return out

    def exp(self) -> "Tensor":
        out = Tensor(data=np.exp(self._data), requires_grad=self.requires_grad)
        if out.requires_grad:

            def grad_exp(grad: "Tensor") -> None:
                self.backward(grad * out._data)

            out._grad_fn = grad_exp
        return out

    def log(self) -> "Tensor":
        out = Tensor(data=np.log(self._data), requires_grad=self.requires_grad)
        if out.requires_grad:

            def grad_log(grad: "Tensor") -> None:
                self.backward(grad / self._data)

            out._grad_fn = grad_log
        return out

    def argmax(self, dim: int = None) -> "Tensor":
        out = Tensor(np.argmax(self._data, axis=dim))
        return out

    def view(self, *shape) -> "Tensor":
        out = Tensor(
            data=np.reshape(self._data, shape), requires_grad=self.requires_grad
        )

        if out.requires_grad:

            def grad_view(grad: "Tensor") -> None:
                self.backward(grad.view(*(self.shape)))

            out._grad_fn = grad_view
        return out

    def squeeze(self, dim: int = None) -> "Tensor":
        out = Tensor(
            data=np.squeeze(self._data, axis=dim), requires_grad=self.requires_grad
        )
        if out.requires_grad:

            def grad_squeeze(grad: "Tensor") -> None:
                self.backward(grad.view(*(self.shape)))

            out._grad_fn = grad_squeeze
        return out

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        def get_dim(dim):
            if dim == dim0:
                return dim1
            elif dim == dim1:
                return dim0
            else:
                return dim

        dims = tuple([get_dim(i) for i in range(self.ndim)])
        out = Tensor(data=self._data.transpose(dims), requires_grad=self.requires_grad)
        if out.requires_grad:

            def grad_transpose(grad: "Tensor") -> None:
                self.backward(grad.transpose(dim1, dim0))

            out._grad_fn = grad_transpose
        return out

    def permute(self, *dims) -> "Tensor":
        out = Tensor(data=self._data.transpose(dims), requires_grad=self.requires_grad)
        if out.requires_grad:

            def grad_permute(grad: "Tensor") -> None:
                self.backward(grad.permute(*np.argsort(dims)))

            out._grad_fn = grad_permute
        return out

    @property
    def T(self):
        return self.permute(*np.arange(self.ndim - 1, -1, -1))

    def unsqueeze(self, dim: int) -> "Tensor":
        out = Tensor(
            data=np.expand_dims(self._data, axis=dim), requires_grad=self.requires_grad
        )
        if out.requires_grad:

            def grad_unsqueeze(grad: "Tensor") -> None:
                self.backward(grad.view(*self.shape))

            out._grad_fn = grad_unsqueeze
        return out

    def sum(self, dim: list = None, keepdims: bool = False) -> "Tensor":
        if isinstance(dim, int):
            dim = [dim]

        if dim is not None:
            dims = list(range(self.ndim))
            dim = [dims[d] for d in dim]

        out = Tensor(
            data=np.sum(
                self._data,
                axis=tuple(dim) if dim is not None else dim,
                keepdims=keepdims,
            ),
            requires_grad=self.requires_grad,
        )
        if out.requires_grad:

            def grad_sum(grad: "Tensor") -> None:
                if out.ndim < self.ndim:
                    expanded_shape = tuple(
                        1 if dim is None or i in dim else self.shape[i]
                        for i in range(self.ndim)
                    )
                    grad = grad.view(*expanded_shape)
                self.backward(grad + np.zeros_like(self._data, dtype=self._dtype))

            out._grad_fn = grad_sum
        return out

    def mean(self, dim: list = None, keepdims: bool = False) -> "Tensor":
        if isinstance(dim, int):
            dim = [dim]

        if dim is not None:
            dims = list(range(self.ndim))
            dim = [dims[d] for d in dim]

        if dim is None:
            size = np.prod(self.shape)
        elif not dim:
            size = 1
        else:
            size = np.prod([self.size(d) for d in dim])
        out = Tensor(
            data=np.mean(
                self._data,
                axis=tuple(dim)
                if (dim is not None) and (not isinstance(dim, int))
                else dim,
                keepdims=keepdims,
            ),
            requires_grad=self.requires_grad,
        )
        if out.requires_grad:

            def grad_mean(grad: "Tensor") -> None:
                if out.ndim < self.ndim:
                    expanded_shape = tuple(
                        1 if dim is None or i in dim else self.shape[i]
                        for i in range(self.ndim)
                    )
                    grad = grad.view(*expanded_shape)
                self.backward(
                    (grad + np.zeros_like(self._data, dtype=self._dtype)) / size
                )

            out._grad_fn = grad_mean
        return out

    def relu(self) -> "Tensor":
        out = Tensor(data=np.maximum(0.0, self._data), requires_grad=self.requires_grad)
        if out.requires_grad:

            def grad_relu(grad: "Tensor") -> None:
                self.backward(grad * (self._data > 0))

            out._grad_fn = grad_relu
        return out

    def softmax(self, dim: int = -1) -> "Tensor":
        dims = list(range(self.ndim))
        dim = dims[dim]
        out_data = self._data - self._data.max(axis=dim, keepdims=True)
        out_data = out_data - np.log(np.exp(out_data).sum(axis=dim, keepdims=True))
        out_data = np.exp(out_data)

        out = Tensor(data=out_data, requires_grad=self.requires_grad)
        if out.requires_grad:

            def grad_softmax(grad: "Tensor") -> None:
                if out_data.shape != grad.shape:
                    raise RuntimeError(
                        "Bad softmax. If you see this error, open issue, please"
                    )

                new_pose = (
                    list(range(dim)) + list(range(dim + 1, out_data.ndim)) + [dim]
                )

                out_data_transposed = out_data.transpose(new_pose)
                out_data_transposed_shape = list(out_data_transposed.shape)
                out_data_formatted = out_data_transposed.reshape(
                    -1, out_data.shape[dim]
                )
                grad_formatted = grad.permute(*new_pose).view(-1, out_data.shape[dim])

                first_part = np.einsum(
                    "bi,bj->bij", out_data_formatted, out_data_formatted
                )
                second_part = np.einsum(
                    "bi,ij->bij",
                    out_data_formatted,
                    np.eye(self.size(dim), self.size(dim), dtype=self._dtype),
                )
                grad_parts = second_part - first_part

                grad_result = Tensor(grad_parts.transpose(0, 2, 1))
                grad_result = (
                    grad_formatted.view(
                        grad_formatted.size(0), 1, grad_formatted.size(1)
                    )
                    * grad_parts.transpose(0, 2, 1)
                ).sum(-1)

                self.backward(
                    grad_result.view(*out_data_transposed_shape).permute(
                        *np.argsort(new_pose)
                    )
                )

            out._grad_fn = grad_softmax
        return out

    def _positive_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(self, x):
        exp = np.exp(x)
        return exp / (exp + 1)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        positive = x >= 0
        negative = ~positive

        result = np.empty_like(x, dtype=self._dtype)
        result[positive] = self._positive_sigmoid(x[positive])
        result[negative] = self._negative_sigmoid(x[negative])
        return result

    def sigmoid(self):
        out_data = self._sigmoid(self._data)

        out = Tensor(data=out_data, requires_grad=self.requires_grad)

        if out.requires_grad:

            def grad_sigmoid(grad: "Tensor") -> None:
                self.backward(grad * out_data * (1.0 - out_data))

            out._grad_fn = grad_sigmoid
        return out

    def __str__(self):
        items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
        return "<%s: {%s}>" % (self.__class__.__name__, ", ".join(items))
