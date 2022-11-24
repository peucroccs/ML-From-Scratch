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

def mean(float_list):
    sum_list = sum(list)
    list_mean = sum_list/len(list)
    return mean

print(dict.to_numpy())


class KMeans:
    def __init__(self, nclusters, max_iter=250):
        self.k = nclusters
        self.max_iter = max_iter

    def fit(self, df):
        self.df = df
        self.centroids = []

        df_array = self.df.to_numpy()
        n_features = len(df_array[:, 0]) + 1

        #initializating centroids
        for k in range(self.k):
            centroid = []
            for index in range(n_features):
                fake_feature = random.choice(df_array[:, index])
                centroid.append(fake_feature)
            self.centroids.append(centroid)
        
        #assigning clusters iterating over all rows and calculating the minimun euclidean distance
        clusters = []
        for row in df_array:
            cluster_distances = []
            for c in self.centroids:
                pd = euclidean_dist(row, c)
                cluster_distances.append(pd)
            min_value = min(cluster_distances)
            index_min_cluster = cluster_distances.index(min_value)
            clusters.append(index_min_cluster)
        clusters_array = np.asarray(clusters, dtype=np.float64)
        clusters_array = clusters_array.reshape(len(df_array), 1)
        df_array = np.hstack((df_array, clusters_array))

        
        i = 0
        while i < self.max_iter:
            #list with "arrays" separeted by clusters
            clusters_list = []
            for k in range(self.k):
                cluster_k = []
                for row in df_array:
                    list_row = list(row)
                    if list_row[n_features] == k:
                        cluster_k.append(list_row)
                clusters_list.append(cluster_k)
            i += 1

KMeans = KMeans(nclusters=3, max_iter=1)
KMeans.fit(dict)
