from sklearn import tree
import pydotplus
'''
    0 : sunny, 1 : windy, 2 : rainy
    1 : yes, 0 : no
    0 : poor, 1: rich
'''

clf = tree.DecisionTreeClassifier(criterion="entropy",min_samples_split=2)

X=[[0,1,1],[0,0,1],[1,1,1],[2,1,0],[2,0,1],[2,1,0],[1,0,0],[1,0,1],[1,1,1],[0,0,1]]
Y=["Cinema","Tennis","Cinema","Cinema","Stay in","Cinema","Cinema","Shopping","Cinema","Tennis"]
clf=clf.fit(X,Y)

featureNames=["Weather","Parents","Money"]
classNames=["Cinema","Shopping","Stay in","Tennis"]
dot_data=tree.export_graphviz(clf, out_file=None,feature_names=featureNames,class_names=classNames)

graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("DecisionTreeSimple.pdf") 

    
print(clf.classes_)

print(clf.predict([[1,0,1]]))
