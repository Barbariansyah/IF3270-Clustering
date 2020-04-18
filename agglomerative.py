import numpy as np


def eucledian(a, b):
    return np.linalg.norm(a-b)


class Agglomerative:
    def __init__(self, linkage, n_clusters):
        linkage_funs = {
            'single': self._single_linkage,
            'complete': self._complete_linkage,
            'avg': self._average_linkage,
            'avg_group': self._average_group_linkage
        }
        self.linkage_fun = linkage_funs[linkage]
        self.n_clusters = n_clusters

    def fit(self, X, verbose=False):
        self.dataset = X.copy()
        self.dataset_size = X.shape[0]
        self.instance_dimension = X.shape[1]

        self.labels = np.zeros(self.dataset_size, dtype=int)
        for i in range(self.dataset_size):
            self.labels[i] = i
        self.cluster_count = self.dataset_size

        self.labels_to_data_map = dict()
        for idx, label in enumerate(self.labels):
            self.labels_to_data_map[label] = [idx]

        while self.cluster_count > self.n_clusters:
            dist_matrix = self._calc_dist_matrix()
            min_dist_idx = np.unravel_index(
                np.argmin(dist_matrix, axis=None), dist_matrix.shape)

            self._merge_cluster(min_dist_idx)
            self._normalize_cluster()
            self._update_labels_map()

            self.cluster_count = np.unique(self.labels).size
            if (verbose):
                print(
                    f"\rCluster count = {self.cluster_count}".ljust(40, " "), end="", flush=True)

        if (verbose):
            print("\rFinished!".ljust(40, " "), flush=True)

    def _calc_dist_matrix(self):
        dist_matrix = np.zeros(
            self.cluster_count * self.cluster_count).reshape(self.cluster_count, self.cluster_count)

        for i in range(self.cluster_count):
            dist_matrix[i, i] = np.inf
            for j in range(i + 1, self.cluster_count):
                dist = self.linkage_fun(i, j)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

    def _merge_cluster(self, min_dist_idx):
        label_i = min_dist_idx[0]
        label_j = min_dist_idx[1]
        unique_labels = np.unique(self.labels)

        min_label = label_i if label_i < label_j else label_j
        for i in range(min_label, -1, -1):
            if not np.all(np.isin(i, unique_labels)):
                min_label = i

        for i in range(self.dataset_size):
            if (self.labels[i] == label_i or self.labels[i] == label_j):
                self.labels[i] = min_label

    def _normalize_cluster(self):
        """
        Normalize labels to be dense
        ex: [0, 0, 1, 0, 3, 4] -> [0, 0, 1, 0, 2, 3] 
        """
        i = 0
        j = 0
        while i < self.cluster_count and j < self.dataset_size:
            if i >= self.labels[j]:
                j += 1
            else:
                i += 1
                if i >= self.labels[j]:
                    continue

                cur_label = self.labels[j]
                while j < self.dataset_size:
                    if self.labels[j] == cur_label:
                        self.labels[j] = i
                        j += 1
                    else:
                        break

    def _update_labels_map(self):
        """
        Assign each data to its label on labels_to_data_map
        """
        self.labels_to_data_map = dict()
        for idx, label in enumerate(self.labels):
            if label in self.labels_to_data_map:
                self.labels_to_data_map[label].append(idx)
            else:
                self.labels_to_data_map[label] = [idx]

    def _single_linkage(self, c1, c2):
        """Return the single linkage distance between two cluster, uses eucledian distance"""
        min_dist = np.inf
        idx_c1 = self.labels_to_data_map[c1]
        idx_c2 = self.labels_to_data_map[c2]
        for i in idx_c1:
            for j in idx_c2:
                dist = eucledian(self.dataset[i], self.dataset[j])
                if (dist < min_dist):
                    min_dist = dist

        return min_dist

    def _complete_linkage(self, c1, c2):
        """Return the complete linkage distance between two cluster, uses eucledian distance"""
        max_dist = -np.inf
        idx_c1 = self.labels_to_data_map[c1]
        idx_c2 = self.labels_to_data_map[c2]
        for i in idx_c1:
            for j in idx_c2:
                dist = eucledian(self.dataset[i], self.dataset[j])
                if (dist > max_dist):
                    max_dist = dist

        return max_dist

    def _average_linkage(self, c1, c2):
        """Return the average linkage distance between two cluster, uses eucledian distance"""
        count = 0
        avg_dist = 0
        total_dist = 0

        idx_c1 = self.labels_to_data_map[c1]
        idx_c2 = self.labels_to_data_map[c2]
        for i in idx_c1:
            for j in idx_c2:
                dist = eucledian(self.dataset[i], self.dataset[j])
                total_dist += dist
                count += 1
        
        avg_dist = float(total_dist) / float(count)

        return avg_dist

    def _average_group_linkage(self, c1, c2):
        """Return the average group linkage distance between two cluster, uses eucledian distance"""
        count_c1 = 0
        count_c2 = 0
        total_dist_c1 = 0
        total_dist_c2 = 0    
        avg_group_dist = 0

        idx_c1 = self.labels_to_data_map[c1]
        idx_c2 = self.labels_to_data_map[c2]
        for i in idx_c1:
            total_dist_c1 += self.dataset[i]
            count_c1 += 1
        
        for j in idx_c2:
            total_dist_c2 += self.dataset[j]
            count_c2 += 1

        avg_group_dist = eucledian(total_dist_c1/count_c1, total_dist_c2/count_c2)

        return avg_group_dist


if __name__ == "__main__":
    X = np.array([[9, 10],
                  [10, 10],
                  [1, 1],
                  [0, 1]])

    aggl = Agglomerative('avg_group', 2)
    aggl.fit(X)
    print("\n")
    print(aggl.labels)
