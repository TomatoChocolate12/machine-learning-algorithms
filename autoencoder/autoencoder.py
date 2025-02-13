import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../'))
from ..mlp.mlp import MLP

class Autoencoder(MLP):
    def __init__(self, learning_rate, optimizer, n_neurons, batch_size, epochs, cost_calc):
        super().__init__(learning_rate, optimizer, n_neurons, batch_size, epochs, "regression", cost_calc)
        self.bottleneck_idx = len(n_neurons) // 2 + 1

    def fit(self, X):
        # print(1)
        super().fit(X, X, activ="linear")
    
    def encode(self, X):
        activations, zs = self.forward_prop(X)
        encoded_output = activations[self.bottleneck_idx]
        # print(activations)
        return encoded_output
    
    def decode(self, encoded_X):
        activations = [encoded_X]
        zs = []
        for w, b in zip(self.weights[self.bottleneck_idx:], self.biases[self.bottleneck_idx:]):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activations.append(self.activ(z))
        
        return activations[-1]
