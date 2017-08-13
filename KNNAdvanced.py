import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import socket

h = .02

# -------- load 80% data -------

iris = load_iris()
X=iris.data[:120] # 80/20 rule
Y=iris.target[:120] # 80/20 rule
n_neighbors = 15
clf = neighbors.KNeighborsClassifier(n_neighbors, weights="distance").fit(X,Y)

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
n_neighbors = 15

# ---------  Plot the results ---------------

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf2 = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf2.fit(X, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()

# --------- accurancy test -----------
predicts=clf.predict(iris.data[-30:])
accurancy=accuracy_score(iris.target[-30:],predicts)*100
print("Accurancy Score : ",accurancy,'%')


