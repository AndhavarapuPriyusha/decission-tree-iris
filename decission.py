import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv("/content/Iris (1).csv")
df.head()

x=df.iloc[:,:4]
y=df.iloc[:,-1]
print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=3)  

clf=DecisionTreeClassifier(random_state=5)

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)

ac

from sklearn import tree
tree.plot_tree(clf)

text_representation=tree.export_text(clf)
print(text_representation)

