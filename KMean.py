import numpy as np
import matplotlib.pyplot as pl
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import socket


# -------- load 80% data -------

iris = load_iris()
X=iris.data[:120] # 80/20 rule
Y=iris.target[:120] # 80/20 rule
kmeans =  KMeans(n_clusters=3, random_state=0).fit(X)


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
kmeans2 = KMeans(n_clusters=3, random_state=0).fit(X)

# ---------  Plot the results ---------------
pl.subplot(121)
pl.title("Real classification")
pl.xlabel(iris.feature_names[c1])
pl.ylabel(iris.feature_names[c2])
pl.scatter(X[:, 0], X[:, 1], marker='o', c=Y)

pl.subplot(122)
pl.xlabel(iris.feature_names[c1])
pl.ylabel(iris.feature_names[c2])
pl.title("Classified using kmeans")
pl.scatter(X[:, 0], X[:, 1], marker='o', c=kmeans2.labels_)




# --------- accurancy test -----------
predicts=kmeans.predict(iris.data[-30:])
accurancy=accuracy_score(iris.target[-30:],predicts)*100
print("Accurancy Score : ",accurancy,'%')


pl.show()
