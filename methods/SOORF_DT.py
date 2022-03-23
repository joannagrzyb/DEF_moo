import numpy as np
# import os
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample
from scipy.stats import mode
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.core.problem import starmap_parallelized_eval
from multiprocessing.pool import Pool
from EnsembleDiversityTests.EnsembleDiversityTests import DiversityTests

from .optimization import Optimization
from .bootstrap_optimization import BootstrapOptimization
from utils_diversity import calc_diversity_measures
from .utilities import shuffle_in_unison
from .decisiontree import DecisionTreeClassifier
import math
import sys
import numpy

# Ustawienia opcji wyświetlania tablic numpy, aby cała zawartość była pokazana:
# numpy.set_printoptions(threshold=sys.maxsize)


class SingleObjectiveOptimizationRandomForest_DecisionTree(BaseEstimator):
    def __init__(self, base_classifier, metric_name="BAC", alpha=0.5, n_classifiers=10, test_size=0.5, objectives=1, p_size=100, predict_decision="MV", bootstrap=False, n_proccess=2, max_depth=None, min_samples_split=2):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.classes = None
        self.test_size = test_size
        self.objectives = objectives
        self.p_size = p_size
        self.selected_features = []
        self.predict_decision = predict_decision
        self.metric_name = metric_name
        self.alpha = alpha
        self.bootstrap = bootstrap
        self.n_proccess = n_proccess
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    # Funkcja, dzięki której można ustawić populację inicjującą, gdzie selected_features_indx to tablica 2-wymiarowa zawierająca indeksy wybranych cech
    def set_init_pop(self, X, selected_features_indx):
        n_features = X.shape[1]
        if selected_features_indx is not None:
            # Ustawienie losowych wartości
            self.initial_population = np.random.random((self.p_size, (self.n_classifiers*n_features)))
            for f_counter, features_model_indx in zip(range(n_features), selected_features_indx):
                for feature_index in features_model_indx:
                    # Ustawienie wartości 0.99, w miejscu gdzie cecha została wybrana, dla każdego modelu
                    self.initial_population[:, feature_index+(f_counter*n_features)] = 0.99

    def partial_fit(self, X, y, classes=None):
        self.X, self.y = X, y
        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(self.y, return_inverse=True)

        self.forest = []
        n_samples = len(y)
        n_features = X.shape[1]
        self.max_features = int(math.sqrt(n_features))
        n_sub_samples = round(n_samples*self.bootstrap)
        tree = DecisionTreeClassifier(max_features=self.max_features, max_depth=self.max_depth, min_samples_split=self.min_samples_split)

        # Bootstrap
        X_b = []
        y_b = []
        if self.bootstrap is True:
            for random_state in range(self.n_classifiers):
                # Prepare bootstrap sample
                Xy_bootstrap = resample(X, y, replace=True, random_state=random_state)
                X_b.append(Xy_bootstrap[0])
                y_b.append(Xy_bootstrap[1])
            # Parallelization - run program on n_proccess (threads)
            # pool = Pool(self.n_proccess)
            # Create optimization problem
            # print(X_b)
            problem = BootstrapOptimization(X, y, X_b, y_b, test_size=self.test_size, estimator=tree, n_features=n_features, n_classifiers=self.n_classifiers, metric_name=self.metric_name, alpha=self.alpha, max_features=self.max_features)
            # Losowa populacja inicjująca
            # self.initial_population = np.random.random((100, (self.n_classifiers*n_features)))
            # Populacja początkowa za pomocą sampling LHS()
            self.initial_population = LHS()

            algorithm = DE(
                pop_size=self.p_size,
                sampling=self.initial_population,
                variant="DE/rand/1/bin",
                CR=0.9,
                dither="vector",
                jitter=False
                )
        else:
            # Parallelization - run program on n_proccess (threads)
            # pool = Pool(self.n_proccess)
            # Create optimization problem
            problem = Optimization(X, y, test_size=self.test_size, estimator=tree, n_features=n_features, n_classifiers=self.n_classifiers, metric_name=self.metric_name, alpha=self.alpha, max_features=self.max_features)

            # Losowa populacja inicjująca
            # self.initial_population = np.random.random((100, (self.n_classifiers*n_features)))
            # Populacja początkowa za pomocą sampling LHS()
            self.initial_population = LHS()

            algorithm = DE(
                pop_size=self.p_size,
                sampling=self.initial_population,
                variant="DE/rand/1/bin",
                CR=0.9,
                dither="vector",
                jitter=False
                )

        res = minimize(problem,
                       algorithm,
                       seed=1,
                       save_history=True,
                       # verbose=False)
                       verbose=True)
        # pool.close()
        self.res_history = res.history

        # F returns all Pareto front solutions (quality) in form [-accuracy]
        self.quality = res.F
        # X returns values of selected features
        # print(res.X)
        # print("Wynik", res.F)
        for result_opt in res.X:
            if result_opt > 0.5:
                feature = True
                self.selected_features.append(feature)
            else:
                feature = False
                self.selected_features.append(feature)

        self.selected_features = np.array_split(self.selected_features, self.n_classifiers)
        # self.selected_features is the vector of selected of features for each model in the ensemble, so bootstrap in this loop ensure different bootstrap data for each model
        # random_state = 1
        for id, sf in enumerate(self.selected_features):
            if self.bootstrap is True:
                X_train = X_b[id]
                y_train = y_b[id]
                candidate = tree.fit(X_train, y_train, selected_features=sf)
                # Add candidate to the ensemble
                self.ensemble.append(candidate)
            else:
                candidate = tree.fit(X, y, selected_features=sf)
                # Add candidate to the ensemble
                self.ensemble.append(candidate)

        # Diversity by DiversityTests
        # predictions = []
        # names = []
        # for mem_ind, member_clf in enumerate(self.ensemble):
        #     predictions.append(member_clf.predict(self.X[:, self.selected_features[mem_ind]]).tolist())
        #     names.append(str(mem_ind))
        # test_class = DiversityTests(predictions, names, self.y)
        # self.diversities = test_class.get_avg_pairwise(print_flag=False)
        # print(self.diversities)

    def fit(self, X, y, classes=None):
        self.ensemble = []
        self.partial_fit(X, y, classes)

    def predict(self, X):
        """ Predict the class of each sample in X. """
        n_samples = X.shape[0]
        n_trees = len(self.ensemble)
        predictions = np.empty([n_trees, n_samples])
        for i in range(n_trees):
            predictions[i] = self.ensemble[i].predict(X)
        return mode(predictions)[0][0]

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble]
        return np.average(probas_, axis=0)

    def calculate_diversity(self):
        '''
        entropy_measure_e: E varies between 0 and 1, where 0 indicates no difference and 1 indicates the highest possible diversity.
        kw - Kohavi-Wolpert variance
        Q-statistic: <-1, 1>
        Q = 0 statistically independent classifiers
        Q < 0 classifiers commit errors on different objects
        Q > 0 classifiers recognize the same objects correctly
        '''
        if len(self.ensemble) > 1:
            # All measures for whole ensemble
            self.entropy_measure_e, self.k0, self.kw, self.disagreement_measure, self.q_statistic_mean = calc_diversity_measures(self.X, self.y, self.ensemble, self.selected_features, p=0.01)

            return(self.entropy_measure_e, self.kw, self.disagreement_measure, self.q_statistic_mean)
