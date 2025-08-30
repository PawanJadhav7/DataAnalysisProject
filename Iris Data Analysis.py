from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

#loading the iris dataset
iris = load_iris()

x = iris.data
y = iris.target

#three flower species (labels): setosa, versicolor, virginica
y_names = iris.target_names

test_ids = np.random.permutation(len(x))

#splitting data and labels into train and test
# keeping last 10 entries for testing, rest for training

x_train = x[test_ids[:-10]]
x_test = x[test_ids[-10:]]

y_train = y[test_ids[:-10]]
y_test = y[test_ids[-10:]]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

pred = clf.predict(x_test)
print(pred)
print(y_test)
print(accuracy_score(y_test, pred) * 100)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(x_train)
X_test = sc_X.transform(x_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=  0.2, random_state = 0)
