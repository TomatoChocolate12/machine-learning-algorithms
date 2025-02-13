import numpy as np

class KMeans:

    def __init__(self, k):
        self.k = k
        self.centres = None
        self.train = None
        self.train_labels = None

    def euclidean(self, x, y):
        return np.linalg.norm(x - y, axis=1)

    def plusplus(self):
        # Initialize centroids using kmeans++ approach
        first = self.train[np.random.randint(0, len(self.train))]
        centres = [first]

        for _ in range(1, self.k):
            dist = np.min([self.euclidean(c, self.train) for c in centres], axis=0)
            dist_sq = dist**2
            prob = dist_sq / np.sum(dist_sq)
            next_centroid = self.train[np.random.choice(len(self.train), p=prob)]
            centres.append(next_centroid)
        self.centres = np.array(centres)

    def fit(self, train):
        self.train = train
        labels = np.zeros(len(train), dtype=int)  # Ensure labels are integers
        itera = 0

        self.plusplus()

        while True:
            prev = np.copy(self.centres)

            for i, example in enumerate(train):
                dist = self.euclidean(example, self.centres)
                labels[i] = np.argmin(dist)

            for cla in range(self.k):
                indexes = np.where(labels == cla)[0]
                if len(indexes) > 0:  # Avoid empty clusters
                    self.centres[cla] = train[indexes].mean(axis=0)

            itera += 1
            if itera > 10000 or np.allclose(prev, self.centres):
                break
        self.train_labels = labels

    def predict(self, test):
        labels = np.zeros(len(test))
        for i, example in enumerate(test):
            dist = self.euclidean(example, self.centres)
            labels[i] = np.argmin(dist)

        return labels

    def getCost(self):
        total_cost = 0
        for cla in range(self.k):
            indexes = np.where(self.train_labels == cla)[0]
            if len(indexes) > 0:
                dist = self.euclidean(self.centres[cla], self.train[indexes])
                total_cost += np.sum(dist**2)  # Sum of squared distances for the cluster
        return total_cost
