import numpy as np
from collections import Counter

class KNN_Classifier:
    
    def __init__(self, k, distance_metric):
        self.k = k
        self.distance_metric = distance_metric
        self.x_train = None
        self.y_train = None

    def euclidean(self, x, y):
        ret = np.square(y) - np.square(x)
        return np.linalg.norm(ret, axis=1)


    def manhattan(self, x, y):
        return np.linalg.norm(np.abs(y - x), axis=1)

    def cosine(self, x, y):
        # if x.ndim == 1:
        #     x = x.reshape(1, -1)
        dot_product = np.dot(y, x.T)
        norm_y = np.linalg.norm(y, axis=1)
        norm_x = np.linalg.norm(x)
        denom = norm_x * norm_y
        # print(len(denom), len(dot_product))
        similarity = dot_product / denom
        return 1 - similarity.T
    
    def compute_distance(self, x):
        if self.distance_metric == "euclidean":
            return self.euclidean(x, self.x_train)
        elif self.distance_metric == "manhattan":
            return self.manhattan(x, self.x_train)
        elif self.distance_metric == "cosine":
            return self.cosine(x, self.x_train)
        else:
            raise ValueError("Unsupported distance metric provided.")
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train


    def predict(self, values):

        # print(values)

        y_pred = []
        for x in values:
            distances = self.compute_distance(x)
            # print(len(distances))
            k_indices = np.argpartition(distances, self.k)[:self.k]
            # print(k_indices)
            k_nearest_labels = self.y_train[k_indices]
            # print(k_nearest_labels)
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            y_pred.append(most_common)
            # print(most_common)

        return np.array(y_pred)


class Performance:

    def __init__(self) -> None:
        pass

    def accuracy(self, y_pred, y_test):
        count = 0
        for a, b in zip(y_pred, y_test):
            if a == b:
                count += 1
        return count / len(y_test)

    def precision(self, y_pred, y_test, classes):
        prec = {'micro': None, 'macro': None}
        precision = []

        for c in classes:
            tp = np.sum((y_test == c) & (y_pred == c))
            fp = np.sum((y_test != c) & (y_pred == c))

            if tp + fp > 0:
                precision.append(tp / (tp + fp))
            else:
                precision.append(0)

        prec['macro'] = np.mean(precision)

        total_tp = np.sum([(y_test == c) & (y_pred == c) for c in classes])
        total_fp = np.sum([(y_test != c) & (y_pred == c) for c in classes])

        if total_tp + total_fp > 0:
            prec['micro'] = total_tp / (total_tp + total_fp)
        else:
            prec['micro'] = 0

        return prec


    def recall(self, y_pred, y_test, classes):
        rec = {'micro': None, 'macro': None}
        recall = []

        for c in classes:
            tp = np.sum((y_test == c) & (y_pred == c))
            fn = np.sum((y_test == c) & (y_pred != c))  # Corrected condition

            if tp + fn > 0:
                recall.append(tp / (tp + fn))
            else:
                recall.append(0)

        rec['macro'] = np.mean(recall)

        total_tp = np.sum([np.sum((y_test == c) & (y_pred == c)) for c in classes])
        total_fn = np.sum([np.sum((y_test == c) & (y_pred != c)) for c in classes])

        if total_tp + total_fn > 0:
            rec['micro'] = total_tp / (total_tp + total_fn)
        else:
            rec['micro'] = 0

        return rec


    def f1score(self, y_pred, y_test, classes):
        f1 = {'micro': None, 'macro': None}
        precision = self.precision(y_pred, y_test, classes)
        recall = self.recall(y_pred, y_test, classes)
        f1score = []

        p = precision['macro']
        r = recall['macro']
        if p + r > 0:
            f1score.append(2 * (p * r) / (p + r))
        else:
            f1score.append(0)

        f1['macro'] = np.mean(f1score)

        prec = precision['micro']
        rec = recall['micro']

        if prec + rec > 0:
            f1['micro'] = 2 * (prec * rec) / (prec + rec)
        else:
            f1['micro'] = 0

        return f1
