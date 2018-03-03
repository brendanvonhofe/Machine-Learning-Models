import numpy as np

class CubicParzenWindowClassifier(object):
    """ Class for a Cubic Parzen Window Classifier

    Args:
        side_length (float/int): Length of side to define cube to search around data point for neighbors:
        train (ndarray): Training dataset with shape [num_samples, num_features+1]. The last column are the labels.

    Attributes:
        d (int): Number of features.
        range (float): Half the side length.
        train (ndarray): Training dataset

    Todo:
        -Error checking
    """
    def __init__(self, side_length, train):
        self.d = train.shape[1]-1
        self.range = side_length/2
        self.train = train
        
    def predict(self, x):
        """ Method to predict which class the sample x belongs to.
        Args:
            x (ndarray): Sample of data.

        Returns:
            int: Encoding of class prediction.
        """
        
        neighbors = self.train
        for i in range(self.d):
            neighbors = neighbors[np.absolute(neighbors[:,i]-x[i]) <= self.range]
        votes = {}
        for neighbor in neighbors:
            c = neighbor[-1]
            if(c not in votes):
                votes[c] = 1
            else:
                votes[c] += 1
                
        if(len(votes)==0):
            return 0
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