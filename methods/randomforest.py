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

    def partial_fit(self, X, y):
        """ Creates a forest of decision trees using a random subset of data and
            features. """
        self.forest = []
        n_samples = len(y)
        n_sub_samples = round(n_samples*self.bootstrap)

        # To jest potrzebne dla drzewa z sklearna
        n_features = X.shape[1]
        self.max_features = int(math.sqrt(n_features))

        # for i in range(self.n_estimators):
        #     shuffle_in_unison(X, y)
        #     X_subset = X[:n_sub_samples]
        #     y_subset = y[:n_sub_samples]
        #
        #     tree = DecisionTreeClassifier(max_features=self.max_features, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        #     # czy tu powinno być clone? - nie da sie z tym clf
        #     # tree.fit(X_subset, y_subset)
        #     # self.forest.append(tree)
        #     # print("TREE", tree.fit(X_subset, y_subset))
        #     candidate = tree.fit(X_subset, y_subset)
        #     # print(tree)
        #     self.forest.append(candidate)

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

        return mode(predictions)[0][0]

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
