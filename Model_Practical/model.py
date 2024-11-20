import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
pima = pd.read_csv(r"C:\Users\LOGESH\Pictures\POAI Python Program\diabetes.csv")
print(pima.head()) 

feature_cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X=pima[feature_cols]
Y=pima.Outcome 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3 ,random_state=1) 
clf = DecisionTreeClassifier(max_depth=2)
clf1=clf.fit(X_train,Y_train)
Y_pred = clf1.predict(X_test) 
print("Accuracy:",metrics.accuracy_score(Y_test,Y_pred))

from matplotlib import pyplot as plt
from sklearn import tree
fig = plt.figure()
tree.plot_tree(clf1)
plt.show()
fig.savefig("decisiontree.png")

