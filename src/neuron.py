import numpy as np


class Neuron(object):

    # Private Functions
    def __init__(self, learning_rate: float, X: np.array, label: np.array):
        self._learning_rate = learning_rate
        self._w = 2 * np.random.random(size=X.shape[-1]) - 1
        # [2 * random() - 1 for i in range(X.shape[-1])]
        self._bias = 2 * np.random.random() - 1
        self._data = X
        self._label = label

    def _update_w_offline(self, x: np.array, err: np.array):
        ''' This function update the weight vector using an offline strategy,
            in other words the weight is updated when a set of instances are
            used for trained the net.
        '''
        self._w = self._w + self._learning_rate * np.dot(err, x)

    def _update_w_online(self, x_n: int, err: int):
        ''' This function update the weight vector using an online strategy, in
            other words the weight is updated for each tested instance.
        '''
        self._w = self._w + self._learning_rate * np.dot(err, x_n)

    def _update_bias(self, err: int):
        self._bias = self._bias + self._learning_rate * err

    def _predict(self, x_n) -> float:
        y_pred = np.dot(x_n, self._w) + self._bias
        # sum([x_i * w_i for x_i, w_i in zip(x_n, self.__w)]) + self.__bias

        return y_pred

    # Public Functions
    def get_w(self):
        return self._w

    def get_bias(self):
        return self._bias

    def get_learning_rate(self):
        return self._learning_rate

    def print_status(self, step):
        print('w:', self._w)
        print('b:', self._bias)
        print('y_pred: {0}'.format(
            np.dot(self._data, np.array(self._w)) + self._bias))
        print('total iterations:', step)
