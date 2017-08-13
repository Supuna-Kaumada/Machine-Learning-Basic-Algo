import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import pydotplus
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import socket

iris = load_iris()
X=iris.data[:120] # 80/20 rule
Y=iris.target[:120] # 80/20 rule
clf = DecisionTreeClassifier(criterion="entropy",random_state=0).fit(X,Y)
plot_step=0.02

# ------- construct the tree for all features -------------

dot_data=tree.export_graphviz(clf, out_file=None,feature_names=iris.feature_names,class_names=iris.target_names)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("DecisionTreeAdvanced.pdf")
print("MSG : Complete decision tree is generated to DecisionTreeAdvanced.pdf")


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
clf2 = DecisionTreeClassifier(min_samples_split=2,criterion="entropy",random_state=0).fit(X,Y)
plot_colors = "bry"
plt.xlabel(iris.feature_names[c1])
plt.ylabel(iris.feature_names[c2])
plt.axis("tight")

# ----------plot the edges ----------

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))

Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

# ---------- Plot the training points ----------

for i, color in zip(range(3), plot_colors):
    idx = np.where(Y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],cmap=plt.cm.Paired)
    plt.axis("tight")
plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()


# ---------- construct the tree for two features ----------

dot_data=tree.export_graphviz(clf2, out_file=None,feature_names=[iris.feature_names[c1],iris.feature_names[c2]],class_names=iris.target_names)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("DecisionTreeAdvancedTwo.pdf")
print("MSG : Decision tree("+iris.feature_names[c1]+","+iris.feature_names[c2]+") is generated to DecisionTreeAdvancedTwo.pdf")

plt.show()

# --------- accurancy test -----------
predicts=clf.predict(iris.data[-30:])
accurancy=accuracy_score(iris.target[-30:],predicts)*100
print("Accurancy Score : ",accurancy,'%')


print()
