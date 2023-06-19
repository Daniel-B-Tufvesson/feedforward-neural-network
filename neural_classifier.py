import random
from collections.abc import Callable
from typing import Any

from neural import BaseNeuralNetwork, Vector


def argmax(values: list[float]) -> int:
    highest = values[0]
    highest_index = 0
    for i, value in enumerate(values):
        if value > highest:
            highest = value
            highest_index = i

    return highest_index


class NeuralClassifier:

    def __init__(self, hidden_layer_layout: list[int], network_factory: Callable[..., BaseNeuralNetwork]):
        self.hidden_layer_layout = hidden_layer_layout
        self.classes = []
        self.network_factory = network_factory
        self.network = None  # type: None | BaseNeuralNetwork

    def classify(self, x_data: Vector):
        predicted = self.network.predict(x_data)
        return self.classes[argmax(predicted)]

    def train(self, x_data: list[Vector], y_data: list[Any], **kwargs):
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
        correct = 0
        for x, gold in zip(x_data, y_data):
            predicted = self.classify(x)
            if predicted == gold:
                correct += 1

        return correct / len(x_data) if len(x_data) > 0 else float('nan')

    def evaluate(self, x_data: list[Vector], y_data: list[Any]) -> dict[str, float]:
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

        print(predicted_total)
        print(correct)

        return {
            'accuracy': global_correct / len(x_data),
            'precision': self.compute_precisions(correct, predicted_total),
            'recall': self.compute_recalls(correct, gold_total)
        }

    @staticmethod
    def compute_precisions(correct: dict[str, int], predicted: dict[str, int]) -> dict[str, float]:
        precisions = {}
        for cls, n_pred in predicted.items():
            precisions[cls] = correct[cls] / n_pred if n_pred != 0 else float('nan')
        return precisions

    @staticmethod
    def compute_recalls(correct: dict[str, int], gold: dict[str, int]) -> dict[str, float]:
        recalls = {}
        for cls, n_gold in gold.items():
            recalls[cls] = correct[cls] / n_gold if n_gold != 0 else float('nan')
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
        import pickle
        with open(filename, mode='rb') as file:
            return pickle.load(file)


def test_training():
    import pprint as pp
    from exclude import ff_nn_1

    print('test_training')
    classifier = NeuralClassifier([10, 10, 10], network_factory=ff_nn_1.FF_NN_1_Network)
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
