#Task: Create two decision trees from the student data
#Prepared for CAP5610 HW2 Task 5

#Import modules
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Import data
stud_df = pd.read_csv('/Users/renitawashburn/PycharmProjects/CAP5620/input/studentst5.csv')
stud_df.head()
print(stud_df)

#split dataset in features and target variable
feature_cols = ['early', 'Finish_hmk', 'senior', 'likes_coffee', 'liked_jedi']
X = stud_df[feature_cols]
y = stud_df.A

#Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=1)

# Create Decision Tree classifer object DT1
clf = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=1, max_features=5)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy Depth 1:",metrics.accuracy_score(y_test, y_pred))

# Create Decision Tree classifer object DT2
clf = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1, max_features=5)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy Depth 2:",metrics.accuracy_score(y_test, y_pred))