##Task 2: Construct SVM
# Load libraries
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt

#Input vectors/data
X = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
#y = x1*x2
y = [1,-1,-1,1]

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create support vector classifier
svc = LinearSVC(C=1000)

# Train model
model = svc.fit(X_std, y)

# Plot data points and color using their class
color = ['black' if c == 0 else 'lightgrey' for c in y]
plt.scatter(X_std[:,0], X_std[:,1], c=color)

# Create the hyperplane
w = svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (svc.intercept_[0]) / w[1]

# Plot the hyperplane
plt.plot(xx, yy)
plt.axis("on"), plt.show();



##Task 5: Plot data points
import matplotlib.pyplot as plt
import numpy as np

# Plot data positive class circles
x = np.array([1,2,2])
y = np.array([1,2,0])
plt.scatter(x, y, marker='o')

# Plot data negative class squares
x = np.array([0,1,0])
y = np.array([0,0,1])
plt.scatter(x, y, marker='s')

# Display Graph
plt.show()

#Construct the weight vector of the maximum margin hyperplane by inspection
#and identify thesupport vectors
#Import SVC libraru
from sklearn.svm import SVC

# define the dataset. Y is used to identify class (pos/neg) based on index in X array
X = np.array([[1, 1],[2, 2], [2, 0],
              [0,0],[1,0],[0,1]])
Y = np.array([0, 0, 0, 1, 1, 1])

# define support vector classifier with linear kernel

clf = SVC(C = 1e5, kernel = 'linear')

# fit the above data in SVC
clf.fit(X, Y)

# plot the decision boundary ,data points,support vector etcv
w = clf.coef_[0]
a = -w[0] / w[1]

xx = np.linspace(0, 12)
yy = a * xx - clf.intercept_[0] / w[1]
y_neg = a * xx - clf.intercept_[0] / w[1] + 1
y_pos = a * xx - clf.intercept_[0] / w[1] - 1
plt.figure(1, figsize=(5, 5) )
plt.plot(xx, yy, 'k',
         label=f"Decision Boundary (y ={w[0]}x1  + {w[1]}x2  {clf.intercept_[0]})")
plt.plot(xx, y_neg, 'b-.',
         label=f"Neg Decision Boundary (-1 ={w[0]}x1  + {w[1]}x2  {clf.intercept_[0]})")
plt.plot(xx, y_pos, 'r-.',
         label=f"Pos Decision Boundary (1 ={w[0]}x1  + {w[1]}x2  {clf.intercept_[0]})")

for i in range(6):
    if (Y[i] == 0):
        plt.scatter(X[i][0], X[i][1], color='red', marker='o', label='negative')
    else:
        plt.scatter(X[i][0], X[i][1], color='green', marker='x', label='positive')
plt.legend()
plt.show()

# calculate margin
print(f'Margin : {2.0 / np.sqrt(np.sum(clf.coef_ ** 2))}')

##Task 6: Plot data points

x_values = [0, -1, 1]
plt.plot(x_values, range(len(x_values)))
plt.show()

##Task 7: Titanic linear, quadratic, and RBF kernels
#Import libraries

import numpy as np
#from sklearn.pipeline import make_pipeline
#from sklearn.feature_selection import SelectKBest
#from sklearn.cross_validation import StratifiedKFold
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import cross_val_score
#from sklearn.cross_validation import train_test_split
import scikitplot as skplt
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#Import data
import pandas as pd
train_df = pd.read_csv('/Users/renitawashburn/PycharmProjects/CAP5620/input/train.csv')
test_df = pd.read_csv('/Users/renitawashburn/PycharmProjects/CAP5620/input/test.csv')

#Drop passenger name and cabin
train_df = train_df.drop(['Name'],  axis=1)
test_df = test_df.drop(['Name'], axis=1)
train_df = train_df.drop(['Cabin'],  axis=1)
test_df = test_df.drop(['Cabin'], axis=1)
train_df = train_df.drop(['Ticket'],  axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

#fill in missing cabin values with most common value
common_value = 'S'
data = [train_df, test_df]
for dataset in data:
   dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

#convert sec to numeric and fill in missing
def clean_sex(train_df):
   try:
      return train_df[0]
   except TypeError:
      return "None"

train_df["Sex"] = train_df.Sex.apply(clean_sex)
categorical_variables = ['Sex', 'Embarked']
for variable in categorical_variables:
   train_df[variable].fillna("Missing", inplace=True)
   discarded = pd.get_dummies(train_df[variable],prefix = variable)
   train_df= pd.concat([train_df, discarded], axis = 1)
   train_df.drop([variable], axis = 1, inplace = True)


def clean_sex(test_df):
   try:
      return test_df[0]
   except TypeError:
      return "None"

test_df["Sex"] = test_df.Sex.apply(clean_sex)
categorical_variables = ['Sex', 'Embarked']
for variable in categorical_variables:
   test_df[variable].fillna("Missing", inplace=True)
   discarded = pd.get_dummies(test_df[variable], prefix=variable)
   test_df = pd.concat([test_df, discarded], axis=1)
   test_df.drop([variable], axis=1, inplace=True)

#fill in missing age and fare with averages
train_df["Age"].fillna (train_df.Age.mean(), inplace = True)
test_df["Age"].fillna (test_df.Age.mean(), inplace = True)

train_df["Fare"].fillna (train_df.Fare.mean(), inplace = True)
test_df["Fare"].fillna (test_df.Fare.mean(), inplace = True)

#represent age as integer
train_df = train_df.round({'Age':0})
test_df = test_df.round({'Age':0})

#train_df
#test_df

#on train data, changed survived value to new variable X_train
#ytrain include the survived value

from sklearn.model_selection import train_test_split
x=train_df.drop(["Survived"],axis=1)
y=train_df["Survived"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#going to use cross validation to test our models
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

#create these functions to allow us the ease of computing relevant scores and plot
def acc_score(model):
    return np.mean(cross_val_score(model,x_train,y_train,cv=k_fold,scoring="accuracy"))

def confusion_matrix_model(model_used):
    cm=confusion_matrix(y_test,model_used.predict(x_test))
    col=["Predicted Dead","Predicted Survived"]
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Dead","Predicted Survived"]
    cm.index=["Actual Dead","Actual Survived"]
    cm[col]=np.around(cm[col].div(cm[col].sum(axis=1),axis=0),decimals=2)
    return cm

def importance_of_features(model):
    features = pd.DataFrame()
    features['feature'] = x_train.columns
    features['importance'] = model.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    return features.plot(kind='barh', figsize=(10,10))

#Linear
SVC_lin=SVC(kernel="linear")
SVC_lin.fit(x_train,y_train)

print("Accuracy Linear: " + str(acc_score(SVC_lin)))
confusion_matrix_model(SVC_lin)

#Quadratic
SVC_quad=SVC(kernel="poly")
SVC_quad.fit(x_train,y_train)

print("Accuracy Quadratic: " + str(acc_score(SVC_lin)))
confusion_matrix_model(SVC_quad)

#RBF Kernel
SVC_rbf=SVC(kernel="rbf")
SVC_rbf.fit(x_train,y_train)

print("Accuracy RBF: " + str(acc_score(SVC_rbf)))
confusion_matrix_model(SVC_rbf)