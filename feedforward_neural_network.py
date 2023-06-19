"""
A feedforward network consisting of units organized into layers.
"""

import random

from neural import BaseNeuralNetwork, Vector, Activation, SigmoidActivation


class NeuralUnit:
    """
    The neuron of the network. It receives an input as a vector x, which components are
    then weighted and summed together. This summed value is then put through an activation
    function, which produces the output y of the unit.
    """

    def __init__(self):
        # The output value.
        self.y = 0.0
        # The input value (sum of weighted inputs).
        self.x = 0.0
        # The weights connecting to the units in the previous layer.
        self.incoming_weights = []  # type: list[float]
        # The error of this unit during training.
        self.error = 0

    def activate(self, x: Vector, activation: Activation) -> float:
        """
        Activate this unit.

        :param x:
        :param activation:
        :return:
        """
        self.x = 0
        for x_, w in zip(x, self.incoming_weights):
            self.x += x_ * w
        self.y = activation.compute(self.x)
        return self.y


class NeuralLayer:
    """
    The neural layer consists of a number of parallel neural units. This is a fully
    connected layer, meaning every unit in this layer connects to every unit in the next.
    """

    def __init__(self, size: int, activation: Activation):
        """
        :param size: the number of units in this layer.
        :param activation: the activation function which the units should use.
        """
        self.size = size
        self.units = [NeuralUnit() for _ in range(size)]
        self.activation = activation

    def set_activations(self, activations: Vector) -> None:
        """
        Set the outputs for each unit in this layer.
        :param activations: the values given as a vector.
        """
        for unit, y in zip(self.units, activations):
            unit.y = y

    def activate(self, x: Vector) -> Vector:
        """
        Activate all units in this layer, i.e. each unit receives an input and
        produces an output.

        :param x: the input for this layer as a vector.
        :return: the resulting output of this layer as a vector.
        """
        return [unit.activate(x, self.activation) for unit in self.units]


class NeuralNetwork(BaseNeuralNetwork):
    """
    A feed-forward neural network.
    """

    def __init__(self):
        self.layers = []  # type: list[NeuralLayer]
        self.compiled = False

    def add_layer(self, size: int) -> None:
        """
        Add a new layer to the network.

        :param size: the number of units in the layer.
        """
        layer = NeuralLayer(size, SigmoidActivation())
        self.layers.append(layer)
        self.compiled = False

    def compile(self):
        """
        Compile the network and generate the connections (weight lists)
        between the nodes.
        """
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]

            # Create weight list for each unit.
            for unit in layer.units:
                unit.incoming_weights = [0.0 for _ in range(prev_layer.size)]

    def predict(self, x: Vector) -> Vector:
        """
        Make a prediction from the input data.

        :param x: the input data as a vector.
        :return: the prediction as a vector.
        """
        activations = x

        # Set the input as the activations in the first layer.
        self.layers[0].set_activations(activations)

        # Propagate forwards.
        for i in range(1, len(self.layers)):
            activations = self.layers[i].activate(activations)

        return activations

    def train(self, x_values: list[Vector], y_values: list[Vector],
              learning_rate: float = 0.1, epochs: int = 5) -> None:
        """
        Train the network with backpropagation.

        :param x_values: a list of vectors which make up the input data.
        :param y_values: a list of vectors which make up the output data.
        :param learning_rate: the learning rate. Usually between 0.0 and 1.0.
        :param epochs: the number of times the network should be trained on the data.
        """

        self.randomize_weights()

        # Train for each epoch.
        for epoch in range(epochs):

            # Train on each example.
            for x, target in zip(x_values, y_values):

                predicted = self.predict(x)

                # Compute the errors.
                self.compute_errors(predicted, target)

                # Adjust the weights based on error.
                self.adjust_weights(learning_rate)

    def compute_errors(self, predicted, target) -> None:
        """
        Compute the error for each unit in the network.

        :param predicted: the prediction produced by the network.
        :param target: the target (what the prediction should be).
        """
        # Compute the output error using the gradient.
        derivative = self.layers[-1].activation.derivative
        for unit, t, p in zip(self.layers[-1].units, target, predicted):
            unit.error = (t - p) * derivative(p)

        # Compute the hidden errors. (exclude first layer.)
        for i in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            derivative = layer.activation.derivative

            # Compute the error for the individual units.
            for j, unit in enumerate(layer.units):
                error = 0
                for next_unit in next_layer.units:
                    error += next_unit.error * next_unit.incoming_weights[j]

                unit.error = error * derivative(unit.x)

    def adjust_weights(self, learning_rate: float) -> None:
        """
        Adjust all weights in the network based on the errors of each unit.

        :param learning_rate: the learning rate (usually between 0.0 and 1.0).
        """
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            # Adjust the weights in each unit.
            for next_unit in next_layer.units:
                for k, unit in enumerate(layer.units):
                    next_unit.incoming_weights[k] += learning_rate * next_unit.error * unit.y

    def randomize_weights(self) -> None:
        """Set each weight to a random value between -1.0 and 1.0."""
        for layer in self.layers:
            for unit in layer.units:
                for i in range(len(unit.incoming_weights)):
                    unit.incoming_weights[i] = random.uniform(-1, 1)

    def save(self, filename: str) -> None:
        """
        Save this network to a file.

        :param filename: the name of the file to save it to.
        """
        import pickle
        with open(filename, mode='wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> "NeuralNetwork":
        """
        Load a network from a file.

        :param filename: the name of the file.
        :return: the neural network.
        """
        import pickle
        with open(filename, mode='rb') as file:
            return pickle.load(file)

