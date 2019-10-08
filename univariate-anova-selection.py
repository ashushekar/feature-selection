"""Univariate Statistics or ANOVA model to select most informative feature to decide the target
Here we will use SelectPercentile to select fixed percentage of features
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

# Use SelectPercentile to select 35% of features
select = SelectPercentile(percentile=35)

# fit and transform training set
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))

# The number of features got reduced to 40.
# Now let us apply logistic regression on this data
X_test_selected = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: {}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("Score with reduced features: {}".format(lr.score(X_test_selected, y_test)))
