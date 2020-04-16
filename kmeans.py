import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.clusters_center = []
    
    def fit(self, X):
        done = False
        self.dataset_size = X.shape[0]
        self.instance_dimension = X.shape[1]
        self.labels = np.zeros(self.dataset_size, dtype=int)
        self.gen_random_clusters_center(X)
        while(not done):
            sum_point, total_point, moved = self.assign_cluster(X)
            if(moved):
                self.update_clusters_center(sum_point, total_point)
            else:
                done = True

    def eucledian(self,a,b):
        return np.linalg.norm(a-b)

    def gen_random_clusters_center(self, X):
        portion = self.dataset_size/self.n_clusters
        for k in range(self.n_clusters):
            self.clusters_center.append(X[random.randint(k*portion,(k+1)*portion-1)])

    def assign_cluster(self, data):
        moved = True
        sum_point = np.zeros([self.n_clusters, self.instance_dimension], dtype=float)
        total_point = np.zeros(self.n_clusters, dtype=float)
        old_labels = self.labels.copy()
        label = 0
        for label_idx, point in enumerate(data):
            distance = np.inf
            for idx, center in enumerate(self.clusters_center):
                new_distance = self.eucledian(center,point)
                if(new_distance<distance):
                    distance = new_distance
                    self.labels[label_idx] = idx
            sum_point[self.labels[label_idx]] += point
            total_point[self.labels[label_idx]] += 1
        if((old_labels==self.labels).all()):
            moved = False
        return sum_point, total_point, moved

    def update_clusters_center(self, sum_point, total_point):
        for idx, sum_e in enumerate(sum_point):
            self.clusters_center[idx] = [x/total_point[idx] for x in sum_e]
        
if __name__ == "__main__":
    X = np.array([[5,3],
     [10,15],
     [15,12],
     [24,10],
     [30,45],
     [85,70],
     [71,80],
     [60,78],
     [55,52],
     [80,91],])

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    print(kmeans.clusters_center)
    print(kmeans.labels)