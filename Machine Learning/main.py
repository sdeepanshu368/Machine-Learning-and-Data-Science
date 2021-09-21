# Regression-> Linear Regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# print(diabetes.keys())  # dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
# print(diabetes.DESCR)

# diabetes_X = diabetes.data
diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-20:]

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-20:]

# Custom example --
# diabetes_X = np.array([[1], [2], [3]])
# diabetes_X_train = diabetes_X
# diabetes_X_test = diabetes_X
# diabetes_Y_train = np.array([3, 2, 4])
# diabetes_Y_test = np.array([3, 2, 4])

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predicted = model.predict(diabetes_X_test)

print('Mean squared error is: ', mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))
print('Weights: ', model.coef_)
print('Intercept: ', model.intercept_)

plt.scatter(diabetes_X_test, diabetes_Y_test)
plt.plot(diabetes_X_test, diabetes_Y_predicted)
plt.show()

# Mean squared error is:  2004.3086353199665
# Weights:  [  -1.16924976 -237.18461486  518.30606657  309.04865826 -763.14121622
#   458.90999325   80.62441437  174.32183366  721.49712065   79.19307944]
# Intercept:  153.05827988224112
# ---------------------------------------------------------------------------------
# Mean squared error is:  2561.3204277283867
# Weights:  [941.43097333]
# Intercept:  153.39713623331698
