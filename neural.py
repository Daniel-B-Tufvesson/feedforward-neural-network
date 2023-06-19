"""
A number of base classes, data structures, and functions for
building neural networks.
"""

import math
from abc import ABC, abstractmethod

Vector = list[float]

SIGMOID_NAME = 'sigmoid'
RELU_NAME = 'relu'


class Activation(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, x: float) -> float:
        pass

    @abstractmethod
    def derivative(self, x: float) -> float:
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
    highest = values[0]
    highest_index = 0
    for i, value in enumerate(values):
        if value > highest:
            highest = value
            highest_index = i

    return highest_index