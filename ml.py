from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(iris.data, iris.target)

print(classifier.predict([[5.1, 3.5, 1.4, 1.5]]))