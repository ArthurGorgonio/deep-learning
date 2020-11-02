import numpy as np

from activation import Activation
from neuron import Neuron


class Perceptron(Neuron):
    def __init__(self, learning_rate: float, X: np.array, Y: np.array,
                 func: str):
        super().__init__(learning_rate, X, Y)
        self._activer = Activation(func)

    def run(self, steps=1001, show=100):
        for step in range(steps):
            cost = 0

            for x_n, y_n in zip(self._data, self._label):
                y_pred = self._predict(x_n)
                y_pred = self._activer.chooser(y_pred)
                diff = y_n - int(y_pred)
                self._update_w_online(x_n, diff)
                self._update_bias(diff)
                cost += diff**2

            if step % show == 0:
                print('step {0}: {1}'.format(step, cost))
        self.print_status(step)
