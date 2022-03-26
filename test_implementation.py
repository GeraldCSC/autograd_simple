import math

from autograd import Node


def test_sigmoid():
    x = Node(2)

    result = 1 / (1 + math.e**-x)

    result.backward()

    true_grad = result.val * (1 - result.val)

    assert math.isclose(x.grad, true_grad)


test_sigmoid()
