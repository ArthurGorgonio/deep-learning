import numpy as np

from perceptron import Perceptron
from sigmoid import Sigmoid

print('Solve Logical Ports using Perceptron!!')
logical_and_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
logical_and_solv = np.array([0, 0, 0, 1]).T

perceptron_and = Perceptron(0.01, logical_and_data, logical_and_solv, 'step')
perceptron_and.run(101, 10)
#
print('Using a sigmoid neuron')
sig = Sigmoid(0.01, logical_and_data, logical_and_solv, 'sigmoid')
sig.run(1001, 100)
