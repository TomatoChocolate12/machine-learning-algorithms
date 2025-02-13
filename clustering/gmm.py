import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from scipy.stats import mode

class GMM:
    def __init__(self, n_clusters, max_iter=100, tolerance=1e-6):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.means = None
        self.covariances = None
        self.weights = None

    def euclidean(self, x, y):
        ret = np.square(y) - np.square(x)
        return np.linalg.norm(ret, axis=1)

    def initialize_parameters(self, X):
        self.train = X
        self.n_samples, self.n_features = self.train.shape
        first = self.train[np.random.randint(0, len(self.train))]
        centres = [first]
        for _ in range(1, self.n_clusters):
            dist = np.min([self.euclidean(c, self.train) for c in centres], axis=0)
            dist_sq = dist**2
            prob = dist_sq / np.sum(dist_sq)
            next_centroid = self.train[np.random.choice(len(self.train), p=prob)]
            centres.append(next_centroid)
        self.means = np.array(centres)
        self.covariances = np.array([np.eye(self.n_features) for _ in range(self.n_clusters)])
        self.weights = np.full(self.n_clusters, 1 / self.n_clusters)

    def expectation_step(self, X):
        likelihood = np.zeros((self.n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            try:
                self.covariances[k] += np.eye(self.covariances[k].shape[0]) * 1e-6
                dist = multivariate_normal(self.means[k], self.covariances[k])
                likelihood[:, k] = dist.pdf(X) * self.weights[k]
            except np.linalg.LinAlgError:
                print(f"Covariance matrix for component {k} is singular. Regularization applied.")
                likelihood[:, k] = multivariate_normal(self.means[k], self.covariances[k], allow_singular=True).pdf(X) * self.weights[k]

        total_likelihood = np.sum(likelihood, axis=1, keepdims=True)
        return likelihood / np.maximum(total_likelihood, 1e-10)

    def maximization_step(self, X, responsibilities):
        resp_sums = np.maximum(responsibilities.sum(axis=0), 1e-10)

        self.weights = resp_sums / self.n_samples
        self.means = np.dot(responsibilities.T, X) / resp_sums[:, np.newaxis]

        for k in range(self.n_clusters):
            diff = X - self.means[k]
            weighted_diff = responsibilities[:, k][:, np.newaxis] * diff
            self.covariances[k] = np.dot(weighted_diff.T, diff) / resp_sums[k]
            self.covariances[k] += np.eye(self.n_features) * 1e-6

    def fit(self, X):
        self.initialize_parameters(X)
        for iteration in range(self.max_iter):
            prev_log_likelihood = self.compute_log_likelihood(self.train)

            responsibilities = self.expectation_step(self.train)
            self.maximization_step(X, responsibilities)

            current_log_likelihood = self.compute_log_likelihood(X)
            if np.abs(current_log_likelihood - prev_log_likelihood) < self.tolerance:
                break

    def predict(self, X):
        responsibilities = self.expectation_step(X)
        return np.argmax(responsibilities, axis=1)

    def get_parameters(self):
        return self.weights, self.means, self.covariances

    def get_responsibilities(self, X):
        return self.expectation_step(X)

    def compute_log_likelihood(self, X):
        n_samples = X.shape[0]
        likelihood = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            self.covariances[k] += np.eye(self.covariances[k].shape[0]) * 1e-6
            dist = multivariate_normal(self.means[k], self.covariances[k])
            likelihood[:, k] = dist.pdf(X) * self.weights[k]

        return np.sum(np.log(np.maximum(np.sum(likelihood, axis=1), 1e-10)))

    def aic_bic(self):
        n_params = self.n_clusters * (self.n_features + self.n_features * (self.n_features + 1) / 2 + 1) - 1
        log_likelihood = self.compute_log_likelihood(self.train)
        return -2 * log_likelihood + n_params * np.log(self.n_samples), -2 * log_likelihood + 2 * n_params

