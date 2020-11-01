import numpy as np

import perceptron
from sigmoid import Sigmoid

print('Solve Logical Ports using Perceptron!!')
print('\nAnd:\n')
logical_and_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
logical_and_solv = np.array([0, 0, 0, 1]).T

perceptron_and = perceptron.Perceptron(0.01, logical_and_data,
                                       logical_and_solv, 'step')
perceptron_and.run()

print('Using a sigmoid neuron')
sig = Sigmoid(0.01, logical_and_data, logical_and_solv, 'sigmoid')
sig.run()
