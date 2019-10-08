"""Model-based feature selection: The feature selection model needs to provide some measure of
importance for each feature, so that they can be ranked by this measure. Deciosn trees and
decision tree-based models provide a feature_importances_ attribute, whcih directly encodes the
importance of each feature. Linear models have coefficents, which can also be used to capture
feature importances by considering the absolute values"""

"""The SelectFromModel class selects all features that have an importance measure of the feature 
greater than the provided threshold"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load and Split the dataset
cancer = load_breast_cancer()

# Get deterministic random number
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))

# Add noise features to the data
# the first 30 features are from the dataset and next 50 are noise
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target,
                                                    test_size=0.5, random_state=0)

select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=2),
                         threshold="median")

select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))

X_test_l1 = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: {}".format(lr.score(X_test, y_test)))
lr.fit(X_train_l1, y_train)
print("Score with reduced features: {}".format(lr.score(X_test_l1, y_test)))

