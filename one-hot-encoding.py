"""Let us check one hot encoding on adult dataset"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("adult.csv", header=None, index_col=False,
                   names=["age", "workclass", "fnlwgt", "education", "educational-num", "marital-status",
                          "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss",
                          "hours-per-week", "native-country", "income"])

# but we will use only few columns
data = data[["age", "workclass", "education", "gender", "hours-per-week", "occupation", "income"]]
print(data.head())

# Let us apply one-hot encoding using pd.get_dummies()
print("Original features:\n{}".format(data.columns))
data_dummies = pd.get_dummies(data)
print("Features after one-hot encoding:\n{}".format(data_dummies.columns))

print(data_dummies.head())

# Now let us do some machine learning stuff on above data
# Note: We have to exclude target column from data
features = data_dummies.loc[:, "age": "occupation_Transport-moving"]
X = features.values
y = data_dummies["income_>50K"].values
print("X.shape: {} and y.shape: {}".format(X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print("Test score: {:.2f}".format(log_reg.score(X_test, y_test)))