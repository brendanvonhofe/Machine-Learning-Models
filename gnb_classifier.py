import numpy as np
import pandas as pd
import math

class GaussianNaiveBayesClassifier(object):
    """ Class for a Gaussian Naive Bayes Classifier
    
    Args:
        train (ndarray): Training dataset with shape [num_samples, num_features+1]. The last column are the labels.

    Attributes:
        n (int): Number of training samples.
        d (int): Number of features.
        class_probs (dict[float]): Dictionary of the prior probabilities of the classes.
        feature_params (dict[float]): Dictionary of class conditional feature parameters for a normal distribution (mean and variance)
    Todo:
        Add error checking.
    """
    def __init__(self, train):
        self.n = train.shape[0]
        self.d = train.shape[1]-1 
        self.class_probs = {}
        for c in np.unique(train[:,-1]):
            self.class_probs[c] = train[train[:,-1]==c].shape[0]/self.n
            
        self.feature_params = {} # feature_params[class][feature index][parameter]

        for c in self.class_probs.keys():
            train_c = train[train[:,-1]==c]
            self.feature_params[c] = []
            for feature_col in train_c[:,:-1].T:
                params = {}
                params['mean'] = np.mean(feature_col)
                params['var'] = np.var(feature_col)
                self.feature_params[c].append(params)
            
    def gaussianProb(self, xi, i, c):
        """ Helper function to calculate the gaussian probability that the data point xi belongs to class c.
        Args:
            xi (float): Data point
            i (int): Column index from which the data point came 
            c (int): Encoding of class

        Returns:
            float: Probability 
        """
        var = self.feature_params[c][i]['var']
        mean = self.feature_params[c][i]['mean']
        scalar = 1 / np.sqrt(2*np.pi*var)
        exp = np.exp((-1) * (((xi-mean)**2)/(2*var)))
        return exp * scalar
            
    def predict(self, x):
        """ Method to predict which class the sample x belongs to.
        Args:
            x (ndarray): Sample of data.

        Returns:
            int: Encoding of class prediction.
        """
        evidence = 0
        for c in self.class_probs.keys():
            probXGivenC = 1
            for i,xi in enumerate(x):
                probXGivenC *= self.gaussianProb(xi,i,c)
            evidence += probXGivenC
        
        probs = {}
        for c in self.class_probs.keys():
            probXGivenC = 1
            for i,xi in enumerate(x):
                probXGivenC *= self.gaussianProb(xi,i,c)
                
            probs[c] = (self.class_probs[c] * probXGivenC) / evidence
            
        highest_prob = max(probs.values())
        prediction = [k for k, v in probs.items() if v == highest_prob][0]
        
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