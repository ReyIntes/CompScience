import random

class KNN:
    def __init__(self, k):
        self.k = k  # Number of nearest neighbors
        self.data = []  # Stores training data
    
    def fit(self, X, y):
        # Store the training data and labels
        self.data = list(zip(X, y))
    
    def predict(self, X_test):
        predictions = []
        for point in X_test:
            neighbors = self.get_neighbors(point)
            labels = [label for _, label in neighbors]
            predictions.append(self.majority_vote(labels))
        return predictions
    
    def get_neighbors(self, point):
        distances = [(self.euclidean_distance(point, data_point), label) for data_point, label in self.data]
        distances.sort(key=lambda x: x[0])  # Sort by distance
        return distances[:self.k]  # Return k nearest neighbors
    
    def majority_vote(self, labels):
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return max(label_counts, key=label_counts.get)
    
    def euclidean_distance(self, point1, point2):
        return sum((point1[i] - point2[i]) ** 2 for i in range(len(point1))) ** 0.5

# Example Usage
X_train = [[2, 3], [3, 4], [5, 6], [8, 8], [9, 10], [1, 1], [2, 2], [6, 5], [7, 7]]
y_train = ['A', 'A', 'B', 'B', 'B', 'A', 'A', 'B', 'B']
X_test = [[4, 5], [7, 6]]

knn = KNN(k=4)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print("Predictions:", predictions)
