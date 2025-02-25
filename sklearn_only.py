from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
tree = DecisionTreeClassifier(random_state=0, max_depth=4).fit(X, y)
plot_tree(tree)
print(tree.score(X, y))
plt.show()
