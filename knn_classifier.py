import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KDTree
from sklearn.neighbors import DistanceMetric

class KNearestNeighborsClassifier(object):
    """ Class for a K-Nearest-Neighbors Classifier

    Args:
        k (int): Number of nearest neighbors to use to classify.
        train (ndarray): Training dataset with shape [num_samples, num_features+1]. The last column are the labels.

    Attributes:
        k (int): Number of nearest neighbors.
        classes (int): List of distinct classes.
        tree (KDTree): Sklearn KDTree to find nearest neighbors.
        train (ndarray): Training dataset

    Todo:
        -Error checking
    """
    def __init__(self, k, train):
        self.k = k
        self.classes = np.unique(train[:,-1])
        self.tree = KDTree(train[:,:-1])
        self.train = train
        
    def predict(self, x):
        """ Method to predict which class the sample x belongs to.
        Args:
            x (ndarray): Sample of data.

        Returns:
            int: Encoding of class prediction.
        """
        dist, ind = self.tree.query([x], k=self.k)
        votes = {}
        for i in ind[0]:
            c = self.train[i,-1]
            if(c not in votes):
                votes[c] = 1
            else:
                votes[c] += 1
        voted = max(votes.values())
        prediction = [k for k, v in votes.items() if v == voted][0]
        return prediction
        
    def evaluate(self, test):
        """ Method to evaluate classifier on a set of test data
        Args:
            test (ndarray): Testing dataset in the shape [num_samples, num_features+1] with the labels in the last column.

        Returns:
            float: Accuracy of the classifier (num_correct/num_samples)

        Todo:
            Add support for other metrics.
        """
        
        n_samples = test.shape[0]
        n_correct = 0
        for sample in test:
            label = sample[-1]
            sample = sample[:-1]
            prediction = self.predict(sample)
            if(prediction == label):
                n_correct += 1
        return n_correct/n_samples          