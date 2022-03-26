from dataclasses import dataclass, field
from math import log
from typing import Any, Callable, List

# TODO: generalize to matrices

"""TODO: make the abstraction cleaner, I think it's
    better to have the grad functions turn into actual functions
    with a forward and backward implementation

so like:
    - probably need to figure out a why to see if we need to compute the grad of some object
        - this way we can just wrap both arguments into nodes, if they aren't already
        but for the args that are integers, constants etc we can just not compute the grads
        for them
        - so prev will have many things(it's operands)
        - terribily inefficient if we don't need grads on some of the objects though since
        - we compute it anyway

    class Exp:
        
        def forward(self, a,b):
            unpack_nodes -> a **b -> packnodes
            store intermediate activations

        def _backward(self, upstream_grad):
            # compute the grad for all it's args
"""


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

    def __rsub__(self, other):
        """x - y will be y.__rsub__(x) if x doesn't implement how to sub y"""
        return self.__mul__(-1).__add__(other)

    def __mul__(self, other):
        result = self.val.__mul__(other)
        result = self.__class__(result)

        result.grad_fns.append(MulGradFn(self, unpack_node(other)))

        if isinstance(other, self.__class__):
            result.grad_fns.append(MulGradFn(other, unpack_node(self)))

        return result

    def __pow__(self, other):
        """self**other"""
        result_unpacked = self.val.__pow__(other)
        result = self.__class__(result_unpacked)

        result.grad_fns.append(ExpGradFn(self, self.val, unpack_node(other)))

        if isinstance(other, self.__class__):
            result.grad_fns.append(
                ExpGradFnR(other, unpack_node(self), result_unpacked)
            )

        return result

    def __rpow__(self, other):
        """other**self"""
        result_unpacked = other.__pow__(self.val)
        result = self.__class__(result_unpacked)

        result.grad_fns.append(
            ExpGradFnR(self, unpack_node(other), result_unpacked)
        )

        if isinstance(other, self.__class__):
            result.grad_fns.append(
                ExpGradFn(other, other.val, unpack_node(self))
            )

        return result

    def __truediv__(self, other):
        """self/other"""
        return self.__mul__(other**-1)

    def __rtruediv__(self, other):
        """other/self"""
        return self.__pow__(-1).__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    # order invariant operations
    __radd__ = __add__
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
    """computes the grad of x**const"""

    prev: Node
    curr_val: float
    exp: float

    def compute_grad(self, upstream):
        local = self.exp * self.curr_val ** (self.exp - 1)
        return upstream * local


@dataclass
class ExpGradFnR:
    """computes the grad of const**x the gradient here is
    log(const) * const**x
    """

    prev: Node
    const: float
    result: float

    def compute_grad(self, upstream):
        local = log(self.const) * self.result
        return upstream * local
