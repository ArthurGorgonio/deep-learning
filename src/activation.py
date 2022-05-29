import numpy as np


class Activation:
    def __init__(self, func):
        self._methods = [
            fu[13:] for fu in dir(self)

            if callable(getattr(self, fu)) and fu.startswith("_Activation__")
        ]

        self.set_activation_func(func)

    def chooser(self, y_pred: float):
        func = getattr(self, self._activation_func,
                       lambda s: "Invalid Function")

        return func(y_pred)

    def __step(self, y_pred: float) -> int:
        return np.where(y_pred > 0, 1, 0)

    def __sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def get_activation_func(self):
        return self._activation_func

    def set_activation_func(self, func):
        if func not in self._methods:
            print(f'The {func} function is not implemented!'
                  f'\nThere are {len(self._methods)}, '
                  f'which are: {self._methods}')
        else:
            self._activation_func = '_Activation__' + func
