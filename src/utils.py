import numpy as np


class Activation:
    def chooser(self, name: str, y_pred: float):
        func_name = '_Activation__' + str(name)
        func = getattr(self, func_name,
                       lambda s: 'invalid!\n just "step" is avaliable')

        return func(y_pred)

    def __step(self, y_pred: float) -> int:
        return np.where(y_pred > 0, 1, 0)

    def __sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))


# TODO Define a singleton pattern here to recovery the same instance when use
#   more than one activation function and/or have more than one object that
#   use this class
