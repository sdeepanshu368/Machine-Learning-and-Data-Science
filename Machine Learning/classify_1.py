# Classification-> K Neighbors Classifier
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
# print(iris.DESCR)  # [0]-> Setosa, [1]-> Versicolour, [2]-> Virginica

features = iris.data
labels = iris.target
# print(features[0], labels[0])

model = KNeighborsClassifier()
model.fit(features, labels)

predicted = model.predict([[5.1, 3.5, 1.4, 0.2]])
print(predicted)
