import numpy as np
import pandas as pd

class MLEClassifier(object):
    """ Class for a Maximum-Likelihood Estimation Classifier 
        using the assumption that the data follows a Multivariate Normal Distribution.

    Args:
        train (ndarray): Training dataset with shape [num_samples, num_features+1]. The last column are the labels.

    Attributes:
        means: Dictionary containing the conditional means of the data given the class.
        covs: Dictionary containing the conditional covariance matrices of the data given the class.
        classes: List of distinct classes.   

    Todo:
        -Add error checking
        -Add support for different assumed prior distributions.
    """
    def __init__(self, train):
        # Calculates the distribution parameters for each of the classes
        
        self.means = {}
        self.covs = {}
        self.classes = np.unique(train[:,-1])
        for c in self.classes:
            class_data = train[train[:,-1]==c]
            self.means[c] = np.mean(class_data[:,:-1], axis=0).reshape(3,1)
            self.covs[c] = np.cov(train[:,:-1].T)
    
    def getMultiGaussianProb(self, x, c):
        """ Helper function to get the gaussian probability that the sample x belongs to class c
        Args:
            x (ndarray): Sample of data in the shape of [num_features, 1]
            c (int): Encoding of a class.

        Returns:
            float: Probability that sample x belongs to class c following a multivariate gaussian distribution.
        """
        
        # Multi-variate Gaussian distribution
        scalar = 1 / ( ((2*np.pi)**(len(self.classes)/2)) * (np.linalg.det(self.covs[c])**(1/2)) )
        exp = np.exp( (-1/2) * (np.matmul(np.matmul((x-self.means[c]).T[0],np.linalg.inv(self.covs[c])),(x-self.means[c]))) )
        prob = scalar * exp
        
        return prob
    
    def predict(self, x):
        """ Method to predict the class of a data sample x
        Args:
            x (ndarray): Sample of data

        Returns:
            int: Encoding of class prediction.
        """
        
        n_features = len(x)
        x = x.reshape(n_features,1)
        
        # Gets probabilities for each class that the sample might belong to
        class_probs = {}
        for c in self.classes:
            class_probs[c] = self.getMultiGaussianProb(x,c)
            
        # We predict that the sample belongs to the class with the highest probability
        highest_prob = max(class_probs.values())
        prediction = [k for k, v in class_probs.items() if v == highest_prob][0]
        
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