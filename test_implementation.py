import math

from autograd import Node


def test_sigmoid():
    x = Node(2)

    result = 1 / (1 + math.e**-x)

    result.backward()

    true_grad = result.val * (1 - result.val)

    assert math.isclose(x.grad, true_grad)

def test_self_add():
    x = Node(2)

    result = x + x

    result.backward()

    assert x.grad == 2
