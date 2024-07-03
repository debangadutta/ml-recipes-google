from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = tts(X, y, test_size=.5)

clf1 = tree.DecisionTreeClassifier()
clf1.fit(X_train, y_train)
predictions1 = clf1.predict(X_test)
# print(predictions)
clf2 = KNeighborsClassifier()
clf2.fit(X_train, y_train)
predictions2 = clf2.predict(X_test)

print(f"Decision tree accuracy: {accuracy_score(y_test, predictions1)}")
print(f"KNN accuracy: {accuracy_score(y_test, predictions2)}")