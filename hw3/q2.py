"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 3
SOLUTION
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def make_data(n_samples, features, k_classes):
    # helper function -- DO NOT MODIFY!!!
    X, y = make_blobs(n_samples, n_features=features, centers=k_classes)
    return X, y

def graph(X, y, centers=None, num_classes=2, title="K Means"):
    # helper function -- DO NOT MODIFY!!!
    assert num_classes >= 2, "need at least 2 clusterrs!"
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k', 'w']
    for i in range(num_classes):
        idx = np.where(y == i)[0]
        plt.scatter(X[idx, 0], X[idx, 1], label=str(i), color=colors[i])
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], marker='*', color='black')
    plt.legend()
    plt.title(title)
    plt.show()

class KNearestNeighbors:

    def __init__(self, k, distance='euclidian'):
        if distance == 'euclidian':
            self.distance = self.euclidian
        assert k > 1, "number of cluster needs to be greater than 1!"
        self.k = k

    def fit(self, X, show_progress=False):
        """
        Q:  implement the K-means algorithm

        Inputs
        - X: data, a np.ndarray with shape (N, D)

        Outputs
        - y: label, a np.array with shape (N, )
             y[i] is the cluster classification of X[i]
        """
        N = X.shape[0]
        D = X.shape[1]

        # initialization step: randomly classify each sample point a cluster
        # prev_y is to see whether our clustering assignment changed or not, if no
        # change ==> prev_y == y, stop the fitting

        centers = ...
        y = np.zeros((N))
        prev_y = np.zeros_like(y)
        stop = False
        i = 1


        while not stop and i <= 1000:

            # maximization step: iterate through each sample point and calculate the distances
            # between the point and the centers and assign the center with the shortest distance
            # to the sample point.
            # (note: for this implementations, use as many nested for-loops as you need, your code
            #        will not be graded for its run-time)
            # hint: np.argmin([0, 3, -5, 1, 3]) ==> 2, make sure to use this function!
            # ==> the lowest value -5 is at index 2!
            for i in range(N):
               ...

            # expectation step: locate the new centers for each cluster
            for i in range(...):
                ...

            if show_progress:
                graph(X, y, centers, num_classes=self.k)

            # update stopping conditions (update i, stop, and prev_y)
            ...

        return y, centers
        
    def performance(self, X, y, centers, lmbda=0.25):
        """
        Q:  calculate the performance of current clustering results. for each sample point, we add each sample point's
            distance to its cluster center. Then, we average all the distances as a performance metrics. The lower the
            average distance the better the performance.

        Inputs
        - X: data, np.ndarray with shape (N, D)
        - y: label of X, np.ndarray with shape (N, )
        - centers: the centers of the clusters AFTER performing K-means

        Outputs
        - inertia: a single float, the average of all the distances from a sample point to their assigned cluster center
        """

        N = X.shape[0]
        total_dist = 0 
        for i in range(N):  # iterate through all the data points...
            ...

        # a performance enhancer -- do not modify
        """vvv DO NOT MODIFY vvv"""
        cluster_distances = []
        for k in range(self.k):
            center = centers[k]
            distances = []
            for i in range(self.k):
                if i != k:
                    d = self.distance(center, centers[i])
                    distances.append(d)
            cluster_distances.append(np.min(distances))

        cluster_distances = np.sort(cluster_distances)
        if lmbda == -1:
            penalized_centroids = 0
        else:
            penalized_centroids = max(1, int(self.k * lmbda))
        avg_cluster_dist = np.mean(cluster_distances[:penalized_centroids])

        avg_dist = total_dist / N
        inertia = avg_dist + 1 / avg_cluster_dist
        """^^^ DO NOT MODIFY ^^^"""
        return inertia
    
    def best_cluster(self, X, trials=5, show_progress=False):
        # do not modify
        results = []
        performances = []
        for trial in range(trials):
            y, centers = self.fit(X)
            performance = self.performance(X, y, centers)
            performances.append(performance)
            results.append((y, centers))
            if show_progress:
                graph(X, y, centers, self.k, title="Trial {}: intertia = {}".format(trial + 1, performance))
        best_performance_idx = np.argmin(performances)
        best_result = results[best_performance_idx]
        return best_result[0], best_result[1], np.min(performances)


    def euclidian(self, p, q):
        """
        Q:  implement the euclidian distance.
            sqrt(sum((p - q) ** 2)).
            Note: we do NOT need the square root since we're only using the distance
            to compare each sample point. That said, please implement the method without
            the outer square-root as this can significantly speed up the run-time.

        >>> p = np.array([0.55009018, 0.59069646])
        >>> q = np.array([0.83546993, 0.09752313])
        >>> euclidian(p, q)     # make sure this number is not square rooted
        0.32466153546773396

        >>> p = np.array([0.44823652, 0.8503188, 0.12845955, 0.40286621])
        >>> q = np.array([0.62330031, 0.6261240, 0.74982219, 0.67129247])
        >>> euclidian(p, q)
        0.5390548134969859

        Inputs
        - p: first sample point (np.array)
        - q: second sample point

        Outputs
        - d: euclidian distance between p and q
        """
        # TODO
        return d

def visualize(n_samples, features, k_classes, show_progress=True):
    X, y = make_data(n_samples, features, k_classes)  # create your own custom data - remember to set features=2 so
                                    # you can graph the plots
    K = k_classes

    # graph the data (uncomment/comment at your own will)
    plt.scatter(X[:, 0], X[:, 1])
    plt.title("Train data")
    plt.show()  # click on "x" to exit out the graph and continue running the program

    knn = KNearestNeighbors(K)
    y_pred, centers, score = knn.best_cluster(X, 5, show_progress=show_progress)    # change show_progress to True if you want to see the trials of KNN

    # uncomment this line if you want to see the very final cluster classification
    graph(X, y_pred, centers, num_classes=K, title="Final Cluster")

    print("Best score: {0}".format(score))


if __name__ == '__main__':
    # visualize(1200, 2, 3)   # you can change these 3 numbers!


    # instructor part only
    # ks = [3, 5, 7, 3]
    # for i, K in zip(["", "1", "2", "3"], ks):
    #     file = f"q2_test_data{i}.npy"
    #     Xy = np.load(file)
    #     X = Xy[:, :-1]
    #     y = Xy[:, -1]
    #     knn = KNearestNeighbors(K)
    #     y_pred, centers, score = knn.best_cluster(X, 10, show_progress=False)
    #     print(file)
    #     print(score)
    pass