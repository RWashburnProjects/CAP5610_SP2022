#Program written for CAP5610 HW2 Task 1


# Import modules
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Import data
train = pd.read_csv('/Users/renitawashburn/PycharmProjects/CAP5620/input/train.csv')
test = pd.read_csv('/Users/renitawashburn/PycharmProjects/CAP5620/input/test.csv')

#1 Preprocess titanic training data
# Get descriptive statistics on data (quartiles, mean, count, minimum and maximum values and the standard deviation)
train.describe()

#Use a pairplot to check for missing values
sns.heatmap(train.isnull(),cbar=False)

#Data Cleaning and Data Drop Process
#Missing Fare and Age are filled in with median
train['Fare'] = train['Fare'].fillna(train['Fare'].dropna().median())
train['Age'] = train['Age'].fillna(train['Age'].dropna().median())
# Change to categoric column to numeric
train.loc[train['Sex']=='male','Sex']=0
train.loc[train['Sex']=='female','Sex']=1
# instead of nan values
train['Embarked']=train['Embarked'].fillna('S')
# Change to categoric column to numeric
train.loc[train['Embarked']=='S','Embarked']=0
train.loc[train['Embarked']=='C','Embarked']=1
train.loc[train['Embarked']=='Q','Embarked']=2

#Drop unnecessary columns
drop_elements = ['Name','Cabin','Ticket']
train = train.drop(drop_elements, axis=1)

train.head()
# which columns we have
train.columns

#2 Select a set of important features
#VarianceThreshold selected because of ease of use and ability to interpret
from sklearn.feature_selection import VarianceThreshold

mdlsel = VarianceThreshold(threshold=0.5)
mdlsel.fit(train)
ix = mdlsel.get_support()
#data1 = mdlsel.transform(train)
data1 = pd.DataFrame(mdlsel.transform(train), columns = train.columns.values[ix])
data1.head()

#3 Plot DT
#from sklearn import tree
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split

target = train['Survived'].values
data_features_names = ['Pclass','SibSp','Parch','Fare','Age']
features = train[data_features_names].values

#Build test and training test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

decision_tree = tree.DecisionTreeClassifier(random_state=1, criterion='entropy', max_depth=5)


decision_tree_ = decision_tree.fit(X_train, y_train)
target_predict = decision_tree_.predict(X_test)

print("Decision tree score: ",accuracy_score(y_test, target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :", mean_squared_error(y_test, target_predict))
print ("R2     :", r2_score(y_test, target_predict))

#Plot
tree.plot_tree(decision_tree)



#Random Forrest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = train['Survived'].values
data_features_names = ['Pclass','SibSp','Parch','Fare','Age']
features = train[data_features_names].values

#Build test and training test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

my_forest = RandomForestClassifier(max_depth=5, n_estimators=500, random_state=1, criterion='entropy')


my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)

print("Random forest score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :", mean_squared_error(y_test,target_predict))
print ("R2     :", r2_score(y_test,target_predict))


#4 Apply five-fold cross validation to extract DT average classification accuracy

from sklearn.model_selection import cross_val_score
scores_dt = cross_val_score(decision_tree_, features, target, cv=5)
print("DT Accuracy Scores (5 fold cv):", scores_dt)

#5 Apply five-fold cross validation to extract RF average classification accuracy

from sklearn.model_selection import cross_val_score
scores_rf = cross_val_score(my_forest, features, target, cv=5)
print("RF Accuracy Scores (5 fold cv):",scores_rf)