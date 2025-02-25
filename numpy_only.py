import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

y_train_r = np.copy(y_train.reshape(-1, 1))
y_test_r = np.copy(y_test.reshape(-1, 1))

train = np.concatenate((X_train, y_train_r), axis=1)
test = np.concatenate((X_test, y_test_r), axis=1)

class CART_Classifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def gini(self, labels):
        _, counts_classes = np.unique(labels, return_counts=True)
        gini = 1 - sum(np.square(counts_classes / labels.size))
        return gini
    
    def loss(self, t1, t2):
        n1 = len(t1)
        n2 = len(t2)
        n = n1 + n2
        return (n1 / n) * self.gini(t1) + (n2 / n) * self.gini(t2)
    
    def best_split(self, X, y):
        best_feature, best_threshold = None, None

    
tree = CART_Classifier()




