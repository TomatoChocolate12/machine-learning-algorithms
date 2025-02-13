import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self, X):
        self.data = X
        self.mean = np.mean(X, axis=0)
        X_centered = self.data - self.mean
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_index]
        sorted_eigenvectors = eigenvectors[:, sorted_index]
        self.components = sorted_eigenvectors[:, :self.n_components]
        self.eigenvalues = sorted_eigenvalues
    
    def transform(self):
        X_centered = self.data - self.mean
        return np.dot(X_centered, self.components)
    
    def inverse_transform(self, X_reduced):
        return np.dot(X_reduced, self.components.T) + self.mean
    
    def checkPCA(self):
        X_reduced = self.transform()
        X_reconstructed = self.inverse_transform(X_reduced)
        reconstruction_error = np.mean(np.square(self.data - X_reconstructed))
        print("Reconstruction Error: ", reconstruction_error)
        return reconstruction_error
