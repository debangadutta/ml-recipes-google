from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = tts(X, y, test_size=.5)

clf1 = ScrappyKNN()
clf1.fit(X_train, y_train)
predictions2 = clf1.predict(X_test)

print(f"ScrappyKNN accuracy: {accuracy_score(y_test, predictions2)}")