import numpy as np
import matplotlib.pyplot as plt

class KDE:
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        self.data = np.array(data)

    def _box_kernel(self, distance):
        return 0.5 if np.abs(distance) <= 1 else 0

    def _gaussian_kernel(self, distance):
        return np.exp(-0.5 * distance**2) / np.sqrt(2 * np.pi)

    def _triangular_kernel(self, distance):
        return max(1 - np.abs(distance), 0)

    def _kernel_function(self, distance):
        if self.kernel == 'box':
            return self._box_kernel(distance)
        elif self.kernel == 'gaussian':
            return self._gaussian_kernel(distance)
        elif self.kernel == 'triangular':
            return self._triangular_kernel(distance)
        else:
            raise ValueError("Unsupported kernel type")

    def predict(self, x):
        x = np.array(x)
        densities = []
        for xi in self.data:
            distance = np.linalg.norm((x - xi) / self.bandwidth)
            densities.append(self._kernel_function(distance))
        return np.sum(densities) / (len(self.data) * self.bandwidth ** len(x))

    def visualize(self, x_range, y_range, grid_size=100):
        if self.data.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")
        
        x_vals = np.linspace(x_range[0], x_range[1], grid_size)
        y_vals = np.linspace(y_range[0], y_range[1], grid_size)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array([[self.predict([x, y]) for x in x_vals] for y in y_vals])

        plt.contourf(X, Y, Z, cmap='viridis')
        # plt.scatter(self.data[:, 0], self.data[:, 1], c='red', s=5)
        plt.colorbar(label='Density')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"KDE with {self.kernel.capitalize()} Kernel")
        plt.savefig(f"../../assignments/5/figures/kde/kde_with_{self.kernel}_kernel.png")
        plt.clf()