import random
from collections.abc import Callable
from typing import Any
import neural
from neural import BaseNeuralNetwork, Vector


class NeuralClassifier:
    """
    This classifier uses an underlying neural network for classifying data. The data are given as vectors.
    The classifier needs to be trained before it can classify. The classes are not pre-set before training,
    but is learned during training.
    """

    def __init__(self, hidden_layer_layout: list[int], network_factory: Callable[..., BaseNeuralNetwork]):
        """
        :param hidden_layer_layout: a list of integers, where each integer is the number of units to add to
        its corresponding hidden layer.
        :param network_factory: a callable which creates and returns a new instance of a neural network.
        """
        self.hidden_layer_layout = hidden_layer_layout
        self.classes = []
        self.network_factory = network_factory
        self.network = None  # type: None | BaseNeuralNetwork

    def classify(self, x_data: Vector) -> Any:
        """
        Classify the input data.

        :param x_data: the input data as a vector to classify.
        :return: the predicted class for the input data.
        """

        # Make sure the network has been created.
        if self.network is None:
            raise RuntimeError('network has not been created. Make sure the classifier is trained '
                               'before calling classify.')

        predicted = self.network.predict(x_data)
        return self.classes[neural.argmax(predicted)]

    def train(self, x_data: list[Vector], y_data: list[Any], **kwargs):
        """
        Train the classifier on the data.

        :param x_data: the input data used for classification.
        :param y_data: the correct classification labels for the input data.
        :param kwargs: arguments passed to the neural network.
        """
        self.classes = list(set(y_data))

        # Construct the network.
        self.network = self.network_factory()
        self.network.add_layer(len(max(x_data, key=lambda x: len(x))))
        for node_count in self.hidden_layer_layout:
            self.network.add_layer(node_count)
        self.network.add_layer(len(self.classes))
        self.network.compile()

        # Vectorize the y data labels.
        def vectorize_class(cls: Any) -> Vector:
            vector = [0 for _ in range(len(self.classes))]
            vector[self.classes.index(cls)] = 1
            return vector

        y_vectorized = [vectorize_class(cls) for cls in y_data]

        # Train network.
        self.network.train(x_data, y_vectorized, **kwargs)

    def accuracy(self, x_data: list[Vector], y_data: list[Any]):
        """
        Compute the accuracy of the classifier. This is the fraction of correct classifications
        in relation to all classifications.

        :param x_data: the in-data to predict with.
        :param y_data: the expected out-data to compare the predictions with.
        :return: the accuracy of this classifier with regard to the data. This as a value between 0.0 and 1.0,
        or 'nan' if x_data is empty.
        """
        correct = 0
        for x, gold in zip(x_data, y_data):
            predicted = self.classify(x)
            if predicted == gold:
                correct += 1

        return correct / len(x_data) if len(x_data) > 0 else float('nan')

    def evaluate(self, x_data: list[Vector], y_data: list[Any]) -> dict[str, float]:
        """
        Evaluate the classifier by computing the accuracy, precision and recall.

        :param x_data: the in-data to predict with.
        :param y_data: the expected out-data to compare the predictions with.
        :return: a dictionary where resulting metrics are added to.
        """
        from collections import Counter
        correct = Counter()
        predicted_total = Counter()
        gold_total = Counter()
        global_correct = 0

        for x, gold in zip(x_data, y_data):
            predicted = self.classify(x)

            predicted_total[predicted] += 1
            gold_total[gold] += 1

            if predicted == gold:
                correct[predicted] += 1
                global_correct += 1

        return {
            'accuracy': global_correct / len(x_data),
            'precision': self.compute_precisions(correct, predicted_total),
            'recall': self.compute_recalls(correct, gold_total)
        }

    def compute_precisions(self, correct: dict[str, int], predicted: dict[str, int]) -> dict[str, float]:
        """
        Compute the precision for each class in the classifier.

        :param correct: the number of correct classifications.
        :param predicted: the number of predictions for each class.
        :return: a dictionary where each class is mapped to its precision (class:precision).
        """
        precisions = {}
        for cls in self.classes:
            precisions[cls] = correct[cls] / predicted[cls] if predicted[cls] != 0 else float('nan')
        return precisions

    def compute_recalls(self, correct: dict[str, int], gold: dict[str, int]) -> dict[str, float]:
        """
        Compute the recall for each class in the classifier.

        :param correct: the number of correct classifications.
        :param gold: the number of instances for each class in the gold data.
        :return: a dictionary where each class is mapped to its recall (class:recall).
        """
        recalls = {}
        for cls in self.classes:
            recalls[cls] = correct[cls] / gold[cls] if gold[cls] != 0 else float('nan')
        return recalls

    def save(self, filename: str) -> None:
        """
        Save this classifier to a file.

        :param filename: the name of the file to save it to.
        """
        import pickle
        with open(filename, mode='wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> "NeuralClassifier":
        """
        Load a neural classifier from file.

        :param filename: the name of the file.
        :return: the loaded neural classifier.
        """
        import pickle
        with open(filename, mode='rb') as file:
            return pickle.load(file)


def test_training():
    import pprint as pp
    from feedforward_neural_network import NeuralNetwork

    print('test_training')
    classifier = NeuralClassifier([10, 10, 10], network_factory=NeuralNetwork)
    classes = ['Airplane', 'Bird', 'Superman']
    train_x = [[random.random() for _ in range(20)] for _ in range(100)]
    train_y = [random.choice(classes) for _ in range(100)]

    print('TRAINING...')
    classifier.train(train_x, train_y, epochs=5)
    print('TRAINING COMPLETE.')

    print('EVALUATION:')
    pp.pprint(classifier.evaluate(train_x, train_y))


if __name__ == '__main__':
    test_training()
