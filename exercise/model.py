import numpy as np
from activation_function.activation_function import ActivationFunction


class model:
    def init_network(self):
        network = {}
        network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        network['b1'] = np.array([0.1, 0.2, 0.3])
        network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        network['b2'] = np.array([0.1, 0.2])
        network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
        network['b3'] = np.array([0.1, 0.2])
        
        return network
    
    def forward(self, network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        af = ActivationFunction()

        a1 = np.dot(x, W1) + b1
        z1 = af.sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        z2 = af.sigmoid(a2)

        a3 = np.dot(z2, W3) + b3
        y = af.identify_function(a3)

        return y
