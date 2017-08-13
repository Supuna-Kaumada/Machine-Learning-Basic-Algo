import numpy as np
import matplotlib.pyplot as pl
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from matplotlib.colors import ListedColormap
import socket

def plot_classification_results(clf, X, y, title):
    # Divide dataset into training and testing parts
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.2)

    # Fit the data with classifier.
    clf.fit(X_train, y_train)

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    h = .02  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.figure()
    pl.xlabel(iris.feature_names[c1])
    pl.ylabel(iris.feature_names[c2])
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    pl.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold)

    y_predicted = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    pl.scatter(X_test[:, 0], X_test[:, 1], c=y_predicted, alpha=0.5, cmap=cmap_bold)
    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())
    pl.title(title)
    return score

# -------- load 80% data -------

iris = load_iris()
X=iris.data[:120] # 80/20 rule
Y=iris.target[:120] # 80/20 rule
clf = GaussianNB().fit(X,Y)

# ------- select user desired 2 features ----------

print("Select 2 features")
print("-"*10)
j=0
for i in iris.feature_names:
    print(str(j)+" - "+i)
    j+=1

print()
inp=list(map(int,input("enter choice : ").strip().split(" ")))
c1,c2=inp[0],inp[1]
    

# ---------- select two features only ----------

X=iris.data[:,[c1,c2]]
Y=iris.target
clf2 = GaussianNB().fit(X,Y)

# ---------  Plot the results ---------------
plot_classification_results(clf, X, Y, "")

pl.show()

# --------- accurancy test -----------
predicts=clf.predict(iris.data[-30:])
accurancy=accuracy_score(iris.target[-30:],predicts)*100
print("Accurancy Score : ",accurancy,'%')


