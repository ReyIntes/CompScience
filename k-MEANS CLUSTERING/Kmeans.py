import random

class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k  # Number of clusters
        self.max_iterations = max_iterations
        self.centroids = []
        self.clusters = []

    def fit(self, data):
        # Randomly initialize centroids from the data points
        self.centroids = random.sample(data, self.k)
        
        for _ in range(self.max_iterations):
            # Assign clusters
            self.clusters = self.assign_clusters(data)
            
            # Compute new centroids
            new_centroids = self.compute_centroids()
            
            # Check for convergence
            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids
    
    def assign_clusters(self, data):
        clusters = [[] for _ in range(self.k)]
        for point in data:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_index = distances.index(min(distances))
            clusters[closest_index].append(point)
        return clusters
    
    def compute_centroids(self):
        new_centroids = []
        for cluster in self.clusters:
            if cluster:
                new_centroids.append(self.mean_point(cluster))
            else:
                new_centroids.append(random.choice(self.centroids))  # Avoid empty cluster issue
        return new_centroids
    
    def mean_point(self, cluster):
        n = len(cluster[0])  # Dimension of the data
        mean = [sum(point[i] for point in cluster) / len(cluster) for i in range(n)]
        return mean
    
    def euclidean_distance(self, point1, point2):
        return sum((point1[i] - point2[i]) ** 2 for i in range(len(point1))) ** 0.5
    
    def predict(self, point):
        distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
        return distances.index(min(distances))

# Example Usage
data = [[2, 3], [3, 4], [5, 6], [8, 8], [9, 10], [1, 1], [2, 2], [6, 5], [7, 7]]
kmeans = KMeans(k=3)
kmeans.fit(data)
print("Final Centroids:", kmeans.centroids)
