import pandas as pd
import numpy as np
import random

dict = pd.DataFrame({
    1: [2, 3, 5, 4, 5],
    2: [3, 2, 2, 3, 2],
    3: [5, 6, 2, 2, 4],
    4: [2, 3, 3, 5, 6],
    5: [2, 1, 4, 4, 3],
    6: [1, 1, 2, 2, 1]
})

def euclidean_dist(vector1, vector2):
    dist_sum = []
    for n in range(len(vector2)):
        vet_dist = (vector1[n] - vector2[n]) ** 2
        dist_sum.append(vet_dist)
    mod = (sum(dist_sum)) ** (1 / 2)
    return mod


print(dict.to_numpy())


class KMeans:
    def __init__(self, nclusters, max_iter=250):
        self.k = nclusters
        self.max_iter = max_iter

    def fit(self, df):
        self.df = df
        self.centroids = []

        df_array = self.df.to_numpy()
        features = len(df_array[:, 0])

        for k in range(self.k):
            centroid = []
            for index in range(features + 1):
                fake_feature = random.choice(df_array[:, index])
                centroid.append(fake_feature)
            self.centroids.append(centroid)
        print(self.centroids)

        clusters = []
        for row in df_array:
            cluster_distances = []
            print(row)
            for c in self.centroids:
                pd = euclidean_dist(row, c)
                cluster_distances.append(pd)
            print(cluster_distances)
            min_value = min(cluster_distances)
            index_min_cluster = cluster_distances.index(min_value)
            clusters.append(index_min_cluster)
        clusters_array = np.asarray(clusters, dtype=np.float64)
        clusters_array = clusters_array.reshape(len(df_array), 1)
        df_array = np.hstack((df_array, clusters_array))
        print(df_array)
