# Python code for the project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

df=pd.read_csv('JSVulnerabilityDataSet.csv')
df=df.iloc[:,4:]

X=df.iloc[:,:-1]
T = df['Vuln']

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA( n_components = 5 ))])
pipe = pipe.fit(X)
X = pipe.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X, T, test_size=0.2,random_state=42)

print("\nRegression")
print("-------------------\n")

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

lr = LinearRegression()
ridge = Ridge()

models = [ lr, ridge]
names = ["Linear Regression", "Ridge Regression"]

for name, model in zip(names, models):
    print(name)   
    model.fit(X_train, t_train)
    train_score = model.score(X_train, t_train)
    test_score = model.score(X_test, t_test)
    print("Train Accuracy: {}, Test Accuracy: {}\n".format(train_score, test_score))

print("\nClassification")
print("-------------------\n")

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier

svm = Pipeline([('scaler', StandardScaler()), 
                (('svc', SVC()) )])
logreg = Pipeline([('scaler', StandardScaler()), 
                ('classifier', LogisticRegression(random_state=0))])
knn = KNeighborsClassifier(n_neighbors=3)
nb = GaussianNB()

models = [ svm, logreg, knn, nb]
names = ["Support Vector Machine", "Logistic Regression", "K-Nearest Neighbours", 
         "Naive Bayes"]

for name, model in zip(names, models):
    print(name)   
    clf = OneVsRestClassifier(model)
    clf.fit(X_train, t_train)
    train_score = clf.score(X_train, t_train)
    test_score = clf.score(X_test, t_test)
    print("Train Accuracy: {}, Test Accuracy: {}\n".format(train_score, test_score))

print("\nTree Learning")
print("-------------------\n")

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

dtclass = DecisionTreeClassifier(random_state=0, max_depth=4)
rfclass = RandomForestClassifier(warm_start=True, oob_score=True)
dtreg = DecisionTreeRegressor(random_state=0)
rfreg = RandomForestRegressor(max_depth=2, random_state=0)

models = [ dtclass, rfclass, dtreg, rfreg]
names = ["Decision Tree Classifier", "Random Forest Classifier", "Decision Tree Regressor", 
        "Random Forest Regressor"]

for name, model in zip(names, models):
    print(name)   
    model.fit(X_train, t_train)
    train_score = model.score(X_train, t_train)
    test_score = model.score(X_test, t_test)
    print("Train Accuracy: {}, Test Accuracy: {}\n".format(train_score, test_score))
