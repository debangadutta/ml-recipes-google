from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random
from scipy.spatial import distance

def euc(a,b):       #a,b -> list of numeric features(a->train, b->test)
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist: 
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = tts(X, y, test_size=.5)

clf1 = ScrappyKNN()
clf1.fit(X_train, y_train)
predictions2 = clf1.predict(X_test)

print(f"ScrappyKNN accuracy: {accuracy_score(y_test, predictions2)}")