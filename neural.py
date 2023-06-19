"""
A number of base classes, data structures, and functions for
building neural networks.
"""

import math
from abc import ABC, abstractmethod

Vector = list[float]

# Activation function names.
SIGMOID_NAME = 'sigmoid'
RELU_NAME = 'relu'


class Activation(ABC):
    """
    An abstract base class for the activation function and its derivative.
    """

    def __init__(self, name: str):
        """
        :param name: the name of this activation.
        """
        self.name = name

    @abstractmethod
    def compute(self, x: float) -> float:
        """
        Compute the activation function given the value x.

        :param x: the value to compute the activation for.
        :return: the activation of x.
        """
        pass

    @abstractmethod
    def derivative(self, x: float) -> float:
        """
        The derivative of the activation function.

        :param x: the value to compute the derivative activation for.
        :return: the derivative activation of x.
        """
        pass


class SigmoidActivation(Activation):
    """
    The sigmoid activation function.
    """

    def __init__(self):
        super().__init__(SIGMOID_NAME)

    def compute(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def derivative(self, x: float) -> float:
        a = self.compute(x)
        return (1 - a) * a


class ReLUActivation(Activation):
    """
    The Rectified Linear Unit (ReLU) activation function.
    """

    def __init__(self):
        super().__init__(RELU_NAME)

    def compute(self, x: float) -> float:
        return max(0.0, x)

    def derivative(self, x: float) -> float:
        return 1 if x > 0 else 0


class BaseNeuralNetwork(ABC):
    """
    An abstract base neural network.
    """

    @abstractmethod
    def add_layer(self, size: int) -> None:
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def predict(self, x: Vector) -> Vector:
        pass

    @abstractmethod
    def train(self, x_values: list[Vector], y_values: list[Vector],
              learning_rate: float = 0.1, epochs: int = 5) -> None:
        pass


def argmax(values: list[float]) -> int:
    """
    Find the index of the element with the largest value.

    :param values: a list of values.
    :return: the index as an integer.
    """
    return max(range(len(values)), key=lambda i: values[i])
