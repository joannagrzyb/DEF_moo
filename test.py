from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from methods.randomforest import RandomForestClassifier_alt

# print(np.sqrt)
digits = load_digits(n_class=2)
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

forest = RandomForestClassifier_alt()
forest.fit(X_train, y_train)

accuracy = forest.score(X_test, y_test)
print('The accuracy was', 100*accuracy, '% on the test data.')

classifications = forest.predict(X_test)
print('The digit at index 0 of X_test was classified as a', classifications[0], '.')
