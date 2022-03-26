from dataclasses import dataclass, field
from typing import Any, Callable, List


def unpack_node(node_or_val):
    if isinstance(node_or_val, Node):
        return node_or_val.val
    else:
        return node_or_val


@dataclass
class Node:
    val: float = 0
    # nodes with empty grad_fns are assumed to be leaves
    grad_fns: List[Any] = field(default_factory=list)
    grad: Any = None

    def __add__(self, other):
        result = self.val.__add__(other)
        result = self.__class__(result)

        result.grad_fns.append(AddGradFn(self))

        # if it's a node
        if isinstance(other, self.__class__):
            result.grad_fns.append(AddGradFn(other))

        return result

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __mul__(self, other):
        result = self.val.__mul__(other)
        result = self.__class__(result)

        result.grad_fns.append(MulGradFn(self, unpack_node(other)))

        if isinstance(other, self.__class__):
            result.grad_fns.append(MulGradFn(other, unpack_node(self)))

        return result

    # def __pow__(self, other):
    # result = self.val.__pow__(other)
    # result = self.__class__(result)

    # result.grad_fns.append(ExpGradFn(self, self.val, unpack_node(other)))

    # if isinstance(other, self.__class__):
    # result.grad_fns.append()

    # return result

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    def _dfs_grad(self, upstream):
        # leaf
        if len(self.grad_fns) == 0:
            if self.grad is None:
                self.grad = upstream
            else:
                self.grad += upstream

        else:
            # DFS
            for grad_fn in self.grad_fns:
                grad_fn.prev._dfs_grad(grad_fn.compute_grad(upstream))

    def backward(self):
        self._dfs_grad(1)


# add GATE backwards is a copy gate
@dataclass
class AddGradFn:
    prev: Node

    def compute_grad(self, upstream):
        return upstream


# mul GATE backwards is a switch gate
@dataclass
class MulGradFn:
    prev: Node
    other_operand: float

    def compute_grad(self, upstream):
        return upstream * self.other_operand


@dataclass
class ExpGradFn:
    prev: Node
    curr_val: float
    exp: float

    def compute_grad(self, upstream):
        local = self.exp * self.curr_val ** (self.exp - 1)
        return upstream * local
