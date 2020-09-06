from random import random

import numpy as np

from utils import Activation


class Perceptron:
    __learning_rate = 1e-2
    __w = []
    __bias = 0
    __data = []
    __label = []
    __activation_func = ''

    # Private Functions
    def __init__(self, learning_rate: float, X: np.array, label: np.array,
                 func: str):
        self.__learning_rate = learning_rate
        self.__w = 2 * np.random.randon(size=X.shape[-1]) - 1
        # [2 * random() - 1 for i in range(X.shape[-1])]
        self.__bias = 2 * np.random.random() - 1
        self.__data = X
        self.__label = label
        self.__activation_func = func
        self.__activer = Activation()

    def __update_w(self, x_n: int, err: int):
        self.__w = self.__w + self.__learning_rate * np.dot(err, x_n)

    def __update_bias(self, err: int):
        self.__bias = self.__bias + self.__learning_rate * err

    def __predict(self, x_n) -> float:
        y_pred = np.dot(x_n, self.__w) + self.__bias
        # sum([x_i * w_i for x_i, w_i in zip(x_n, self.__w)]) + self.__bias

        return y_pred

    # Public Functions
    def get_w(self):
        return self.__w

    def get_bias(self):
        return self.__bias

    def get_learning_rate(self):
        return self.__learning_rate

    def run(self):
        step = 0

        while True:
            step += 1
            err = 0

            for x_n, y_n in zip(self.__data, self.__label):
                y_pred = self.__predict(x_n)
                y_pred = self.__activer.chooser(self.__activation_func, y_pred)
                diff = y_n - y_pred
                self.__update_w(x_n, diff)
                self.__update_bias(diff)
                err += diff**2

            if err == 0:
                break

        print('w:', self.__w)
        print('b:', self.__bias)
        print('y_pred: {0}'.format(
            np.dot(self.__data, np.array(self.__w)) + self.__bias))
        print('total iterations:', step)


# TODO
# clean the come
# do a base class for all Neurons and Neural Networs import basic functions
