from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

#import dataset
iris = load_iris()
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
# print(iris.target[0])       #0->setosa

#splitting dataset
test_idx = [0, 50, 100]     #1st data of every type of flower

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#train classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#predict
print(f"Test target: {test_target}")
print(f"Predicted target: {clf.predict(test_data)}")

#visualize the tree
import pydot
from six import StringIO
dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph[0].write_pdf("iris.pdf")