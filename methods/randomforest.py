from __future__ import division
import numpy as np
from scipy.stats import mode
from .utilities import shuffle_in_unison
from .decisiontree import DecisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.base import BaseEstimator, clone


class RandomForestClassifier_alt(BaseEstimator):
    """ A random forest classifier.

    A random forest is a collection of decision trees that vote on a
    classification decision. Each tree is trained with a subset of the data and
    features.
    """

    def __init__(self, n_estimators=10, max_features=0, max_depth=None,
                 min_samples_split=2, bootstrap=0.9, bootstrapping=True):
        # (self, n_estimators=32, max_features=np.sqrt, max_depth=10,
        # min_samples_split=2, bootstrap=0.9):
        """
        Args:
            n_estimators: The number of decision trees in the forest.
            max_features: Controls the number of features to randomly consider
                at each split.
            max_depth: The maximum number of levels that the tree can grow
                downwards before forcefully becoming a leaf.
            min_samples_split: The minimum number of samples needed at a node to
                justify a new node split.
            bootstrap: The fraction of randomly choosen data to fit each tree on.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.bootstrapping = bootstrapping
        self.forest = []

    def partial_fit(self, X, y, classes=None):
        """ Creates a forest of decision trees using a random subset of data and
            features. """
        self.X, self.y = X, y
        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(self.y, return_inverse=True)

        self.forest = []
        self.selected_features_indx = []
        n_samples = len(y)
        n_sub_samples = round(n_samples*self.bootstrap)

        n_features = X.shape[1]
        self.max_features = int(math.sqrt(n_features))

        n_sub_samples = round(n_samples*self.bootstrap)

        for i in range(self.n_estimators):
            # print(X.shape[1])
            shuffle_in_unison(X, y)
            tree = DecisionTreeClassifier(max_features=self.max_features, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            if self.bootstrapping is True:
                X_subset = X[:n_sub_samples]
                y_subset = y[:n_sub_samples]
                candidate = tree.fit(X_subset, y_subset)
            else:
                candidate = tree.fit(X, y)

            self.forest.append(candidate)
            self.selected_features_indx.append(tree.selected_features_indx)

    def fit(self, X, y):
        self.forest = []
        self.partial_fit(X, y)
        return self

    def predict(self, X):
        """ Predict the class of each sample in X. """
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        # print(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in range(n_trees):
            predictions[i] = self.forest[i].predict(X)
        # print(predictions)
        # print(mode(predictions)[0][0])

        return mode(predictions)[0][0]

    # def ensemble_support_matrix(self, X):
    #     # Ensemble support matrix
    #     return np.array([member_clf.predict_proba(X) for member_clf in self.forest])
    #
    # def predict(self, X):
    #     # Prediction based on the Average Support Vectors
    #     # ens_sup_matrix = self.ensemble_support_matrix(X)
    #     # average_support = np.mean(ens_sup_matrix, axis=0)
    #     # prediction = np.argmax(average_support, axis=1)
    #     # return self.classes_[prediction]
    #     # majority Voting
    #     predictions = np.array([member_clf.predict(X) for member_clf in self.forest])
    #     prediction = np.squeeze(mode(predictions, axis=0)[0])
    #     return self.classes_[prediction]
    #
    # def predict_proba(self, X):
    #     probas_ = [clf.predict_proba(X) for clf in self.forest]
    #     return np.average(probas_, axis=0)

    def score(self, X, y):
        """ Return the accuracy of the prediction of X compared to y. """
        y_predict = self.predict(X)
        n_samples = len(y)
        correct = 0
        for i in range(n_samples):
            if y_predict[i] == y[i]:
                correct = correct + 1
        accuracy = correct/n_samples
        return accuracy
