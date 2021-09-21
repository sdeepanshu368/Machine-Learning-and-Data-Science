# Classification-> Logistic Regression
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris['DESCR'])
# print(iris['data'])
# print(iris['data'].shape)
# print(iris['target'])

X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(int)  # use np.int_, np.int32, np.int64 or int only

clf = LogisticRegression()
clf.fit(X, y)
example = clf.predict(([[2.6]]))  # [0]-> False, [1]-> True
print(example)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = clf.predict_proba(X_new)
# print(y_prob)
plt.plot(X_new, y_prob[:, 1], "g-", label="virginica")
plt.show()
