#https://www.kaggle.com/prashant111/logistic-regression-classifier-tutorial

###POLARITY ANALYSIS
#Read in CSV
import pandas as pd
import sklearn.model_selection

df = pd.read_csv('/Volumes/USBPM951/nursingMLedited.csv')
#df.head()


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
#%matplotlib inline

#Drop rows were selftext is nan. Drop removed by category column
df = df[df['selftext'].notna()]
df = df.drop('removed_by_category', 1)

#Declare feature vector and target variable. Label is polarity
X = df.drop(['emotion', 'selftextc','author','created_utc','id','selftext','title','newdate_x',
                     'created_date','link_date','Date_x','day_of_week','emotion_final','label', 'compound', 'pos',
			 		'neu', 'neg', 'fear', 'sad', 'angry', 'happy', 'suprise'], axis=1)
y = df['label']

# split X and y into training and testing sets
#test size  = proportion of the dataset to include in the test split
#random_state = Controls the shuffling applied to the data before applying the split.
#set random_state to some integer to allow for reproducable results
#Option1 for splitting data: hold out
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# check the shape of X_train and X_test
#X_train.shape, X_test.shape

#Option 2: kfold
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

print(skf)

for train_index, test_index in skf.split(X,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


#Scaling
#scaling features to lie between a given minimum and maximum value, between zero and one
cols = X_train.columns
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.describe()

#X_train dataset is now ready to be fed intro Logistic Regression classifier
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression

# instantiate the model
#Solver: For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’
#and ‘lbfgs’ handle multinomial loss; newton-cg only supports l2 or none penalty
logreg = LogisticRegression(penalty='none', solver='newton-cg', multi_class='multinomial', warm_start=True)

# fit the model
logreg.fit(X_train, y_train)
#predit results
y_pred_test = logreg.predict(X_test)
y_pred_test

# probability of getting output as 0 - label
X_test_predprod = logreg.predict_proba(X_test)
X_test_predprod
#add predict results & probabilities to df
# create an Empty DataFrame object
#df_logreg = pd.DataFrame()
#df_logreg['pred_result'] = y_pred_test
#df_logreg['pred_prod'] = logreg.predict_proba(X_test)

#check precision, recall,f1-score using classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_test))



#feature_importance = abs(logreg.coef_[0])
#print(feature_importance)
# get Feature importance
from matplotlib import pyplot
#importance = logreg.coef_
importance = abs(logreg.coef_[0])
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

(pd.Series(importance, index=X.columns)
   #.nlargest(4)
   .plot(kind='barh'))

# create an Empty DataFrame object to hold LogReg Accuracy scores
#df_logregacc = pd.DataFrame()
#Check accuracy
#y_test are the true class labels and y_pred_test are the predicted class labels in the test-set.
from sklearn.metrics import accuracy_score

#y_test are the true class labels and y_pred_test are the predicted class labels in the test-set.
print('LogReg Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
#df_logregacc['model_accuracy'] = accuracy_score(y_test, y_pred_test)
#compare the train-set and test-set accuracy to check for overfitting
y_pred_train = logreg.predict(X_train)
y_pred_train

print('LogReg Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#df_logregacc['trainset_accuracy'] = accuracy_score(y_train, y_pred_train)
#Check for over & underfitting
# print the scores on training and test set
print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
#df_logregacc['trainset_score'] = logreg.score(X_train, y_train)
#df_logregacc['testset_score'] = logreg.score(X_test, y_test)

# fit the Logsitic Regression model with C=100

# instantiate the model
#logreg100 = LogisticRegression(C=100, solver='newton-cg', random_state=0)

# fit the model
#logreg100.fit(X_train, y_train)
# print the scores on training and test set
#print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))
#print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))

#Null accuracy is the accuracy that could be achieved by always predicting the most frequent class.
# check class distribution in test set
#Find the occurences of most frequent class is n. calculate null accuracy by dividing n by total number of occurences.

#y_test.value_counts()

# check null accuracy score
#null_accuracy = (n/(total))
null_accuracy = (4273/(4273+2912+2216))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

#confusion matrix is a tool for summarizing the performance of a classification algorithm
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap
    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title('Polarity Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Polarity Category')
    ax.set_ylabel('Actual Polarity Category ');

    ## Ticket labels - List must be in alphabetical order
    #CHECK CHECK
    ax.xaxis.set_ticklabels(['neg','neu', 'pos'])
    ax.yaxis.set_ticklabels(['neg','neu', 'pos'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()
#cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                # index=['Predict Positive:1', 'Predict Negative:0'])

#sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#Classification report is another way to evaluate the classification model performance.
# It displays the precision, recall, f1 and support scores for the model.
#from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred_test))

#TP = cm[0,0]
#TN = cm[1,1]
#FP = cm[0,1]
#FN = cm[1,0]

# print classification accuracy
#classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
#print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# print classification error
#classification_error = (FP + FN) / float(TP + TN + FP + FN)
#print('Classification error : {0:0.4f}'.format(classification_error))

#Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes
# print precision score
#precision = TP / float(TP + FP)
#print('Precision : {0:0.4f}'.format(precision))

#Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes
#recall = TP / float(TP + FN)
#print('Recall or Sensitivity : {0:0.4f}'.format(recall))

# print predicted probabilities of three classes

y_pred_prob = logreg.predict_proba(X_test)
y_pred_prob
# store the probabilities in dataframe
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob_of-neg', 'Prob_of _eu', 'Prob_of_pos'])
#PRINT LATER
y_pred_prob_df

# Applying 5-Fold Cross Validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# compute Average cross-validation score
print('Average cross-validation score: {:.4f}'.format(scores.mean()))



####Imbalanced Multiclass Classification
# summarize the class distribution
from collections import Counter
target = y.values
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

from collections import Counter
from numpy import mean
from numpy import std
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
	# evaluate model
	#n_jobs = -1 means using all processors
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define the reference model
#model = DummyClassifier(strategy='most_frequent')

# evaluate the model
#scores = evaluate_model(X, y, model)
# summarize performance
#print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

##Test multiple models
# define models to test
def get_models():
	models, names = list(), list()
	# LDA
	models.append(LinearDiscriminantAnalysis())
	names.append('LDA')
	# SVM
	models.append(LinearSVC())
	names.append('SVM')
	# Bagging
	models.append(BaggingClassifier(n_estimators=1000))
	names.append('BAG')
	# RF
	models.append(RandomForestClassifier(n_estimators=1000))
	names.append('RF')
	# ET
	models.append(ExtraTreesClassifier(n_estimators=1000))
	names.append('ET')
	return models, names

#enumerate the list of models in turn and evaluate each, storing the scores for later evaluation
# define models
models, names = get_models()
results = list()
# evaluate each model
for i in range(len(models)):
	# evaluate the model and store results
	scores = evaluate_model(X, y, models[i])
	results.append(scores)
	# summarize performance
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))

# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()