import numpy as np

from activation import Activation
from neuron import Neuron


class Sigmoid(Neuron):
    def __init__(self, learning_rate: float, X: np.array, Y: np.array,
                 func: str):
        super().__init__(learning_rate, X, Y)
        self._activer = Activation(func)

    def run(self, steps=1001, show=100):

        for step in range(steps):
            z = np.dot(self._data, self._w.T) + self._bias
            y_pred = self._activer.chooser(z)
            error = self._label - y_pred

            self._update_w_offline(self._data, error.T)
            self._update_bias(error.sum())

            if step % show == 0:
                cost = np.mean(-self._label * np.log(y_pred) -
                               (1 - self._label) * np.log(1 - y_pred))
                print('step {0}: {1}'.format(step, cost))
        self.print_status(step)
