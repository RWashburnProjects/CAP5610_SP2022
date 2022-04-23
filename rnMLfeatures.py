import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
import seaborn as sns

#Read in CSV
df = pd.read_csv('/Volumes/USBPM951/MLeditedemotionV2.csv')

#Drop rows were selftext is nan. Drop removed by category column
df = df[df['selftext'].notna()]
df = df.drop('removed_by_category', 1)

#Feature creation
import re
#Title word count
#wc_title = len(re.findall(r'\w+', 'title'))
wc_title = df['title'].str.lower().str.split()
wc_title2 = wc_title.apply(len)
wc_selftext = df['selftextc'].str.lower().str.split()
wc_selftext2 = wc_selftext.apply(len)

#res = len(test_string.split())
#Self text word count
#wc_selftext= len(re.findall(r'\w+', 'selftextc'))
df['wc_title2']  = wc_title2
df['wc_selftext2'] = wc_selftext2

#Create df for feature selection analysis
df2 = df.drop(['emotion', 'selftextc','author','created_utc','id','selftext','title','newdate_x',
                     'created_date','link_date','Date_x','day_of_week','emotion_final','label',
             'emotion','KEY'], axis=1)
df2.head()

#Declare feature vector and target variable.
#Predicting Number of comments
X = df.drop(['emotion', 'selftextc','author','created_utc','id','selftext','title','newdate_x',
                     'created_date','link_date','Date_x','day_of_week','emotion_final','label',
             'emotion','KEY', 'num_comments'], axis=1)
y = df['num_comments']

##FEATURE SELECTION
#Correlation matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
corr_matrix = df2.corr(method="spearman")
figure(figsize=(5, 5))
plt.rcParams.update({'font.size': 12})
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
#figure(figsize=(1, 1))
plt.title("spearman correlation")
corr_matrix


import ppscore as pps
#ppscore library is an asymmetric, data-type-agnostic score that can detect linear or non-linear
#relationships between two columns. The score ranges from 0 (no predictive power) to 1 (perfect predictive power)
pps_matrix = pps.matrix(df)
pps_matrix
pps_matrix.to_csv('/Volumes/USBPM951/ppsmatrix.csv')
pps_y = pps.predictors(df2, 'num_comments', sample = None, random_seed = 100)
pps_y

#Using Pearson Correlation (Red)
plt.figure(figsize=(12,10))
cor = df2.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Correlation with output variable
cor_target = abs(cor['num_comments'])
cor_target
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.2]
relevant_features

#Feature analysis to address multicollinearity problem, which refers to a situation
# in which two or more explanatory variables in a multiple regression model are highly linearly related
#https://towardsdatascience.com/machine-learning-with-python-regression-complete-tutorial-47268e546cea
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition

feature_names = X.columns
## p-value
selector = feature_selection.SelectKBest(score_func=
                                         feature_selection.f_regression, k=10).fit(X, y)
pvalue_selected_features = feature_names[selector.get_support()]

## regularization
selector = feature_selection.SelectFromModel(estimator=
                                             linear_model.Ridge(alpha=1.0, fit_intercept=True),
                                             max_features=10).fit(X, y)
regularization_selected_features = feature_names[selector.get_support()]

## plot
dtf_features = pd.DataFrame({"features":feature_names})
dtf_features["p_value"] = dtf_features["features"].apply(lambda x: "p_value" if x in pvalue_selected_features else "")
dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in pvalue_selected_features else 0)
dtf_features["regularization"] = dtf_features["features"].apply(lambda x: "regularization" if x in regularization_selected_features else "")
dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in regularization_selected_features else 0)
dtf_features["method"] = dtf_features[["p_value","regularization"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
dtf_features["method"] = dtf_features["method"].apply(lambda x: "both" if len(x.split()) == 2 else x)
sns.barplot(y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False), dodge=False)

##Feature selection: Gradiant Boosting
## call model
model = ensemble.GradientBoostingRegressor()
## Importance
model.fit(X, y)
importances = model.feature_importances_
## Put in a pandas dtf
dtf_importances = pd.DataFrame({"IMPORTANCE": importances,
                                "VARIABLE": feature_names}).sort_values("IMPORTANCE",
                                                                        ascending=False)
dtf_importances['cumsum'] =dtf_importances['IMPORTANCE'].cumsum(axis=0)
dtf_importances = dtf_importances.set_index("VARIABLE")

## Plot
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
fig.suptitle("Features Importance", fontsize=20)
ax[0].title.set_text('variables')
dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(
    kind="barh", legend=False, ax=ax[0]).grid(axis="x")
ax[0].set(ylabel="")
ax[1].title.set_text('cumulative')
dtf_importances[["cumsum"]].plot(kind="line", linewidth=4,
                                 legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)),
          xticklabels=dtf_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
plt.show()

## split data
dtf_train, dtf_test = model_selection.train_test_split(df2, test_size=0.3)


#Limit to features selected
X_names = ['over_18', 'sad', 'pc_tot_pedpt_covid', 'st_count', 'pc_prevday_adm_all', 'tot_adpt_covid_x', 'deaths_covid_x',
          'pc_tot_pthosp', 'pc_tot_adpt_covid', 'tot_pthosp_x', 'score']
X_train = dtf_train[X_names]
X_train.columns = X_names
#X_train = dtf_train.drop('num_comments', axis=1).values
y_train = dtf_train["num_comments"]
X_test = dtf_test[X_names]
X_test.columns = X_names
#X_test = dtf_test.drop('num_comments', axis=1).values
y_test = dtf_test["num_comments"]

#Scale training datasets
from sklearn.preprocessing import MinMaxScaler
#constrain the range of values to be between 0 and 1 because of previous error
scaler = MinMaxScaler().fit(X_train)
#scale Training
X_train = scaler.transform(X_train)

#Scale testing
X_test = scaler.transform(X_test)
## scale Ys
#scalerY = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
scalery = MinMaxScaler().fit(y_train)
y_train = scalery.transform(y_train.reshape(-1,1))
y_test = scalery.transform(y_test.reshape(-1,1))


## call model

model = linear_model.LinearRegression()
## K fold validation
scores = []
cv = model_selection.KFold(n_splits=5, shuffle=True)
fig = plt.figure()
i = 1
for train, test in cv.split(X_train, y_train):
    prediction = model.fit(X_train[train],
                 y_train[train]).predict(X_train[test])
    true = y_train[test]
    score = metrics.r2_score(true, prediction)
    scores.append(score)
    plt.scatter(prediction, true, lw=2, alpha=0.3,
                label='Fold %d (R2 = %0.2f)' % (i,score))
    i = i+1
plt.plot([min(y_train),max(y_train)], [min(y_train),max(y_train)],
         linestyle='--', lw=2, color='black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('K-Fold Validation_LinearRegression')
plt.legend()
plt.show()

## call model
model2 = ensemble.GradientBoostingRegressor(random_state=0)
## K fold validation
scores2 = []
cv2 = model_selection.KFold(n_splits=5, shuffle=True)
fig = plt.figure()
i = 1
for train, test in cv2.split(X_train, y_train):
    prediction = model2.fit(X_train[train],
                 y_train[train]).predict(X_train[test])
    true = y_train[test]
    score2 = metrics.r2_score(true, prediction)
    scores2.append(score2)
    plt.scatter(prediction, true, lw=2, alpha=0.3,
                label='Fold %d (R2 = %0.2f)' % (i,score))
    i = i+1
plt.plot([min(y_train),max(y_train)], [min(y_train),max(y_train)],
         linestyle='--', lw=2, color='black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('K-Fold Validation_GradiantBoosting')
plt.legend()
plt.show()


## train Linear Regression
model.fit(X_train, y_train)
## test
predicted = model.predict(X_test)
LRscore = model.score(X_test, y_test)
LRscore

from sklearn.metrics import mean_squared_error, r2_score
# The coefficients
print("Coefficients: \n", model.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, predicted))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, predicted))







## train Gradient Boosting
model2.fit(X_train, y_train.ravel())
## test
predicted2 = model2.predict(X_test)
BGscore = model2.score(X_test, y_test)
BGscore


#Export df to Excel
df2 = df1.copy()
    with pd.ExcelWriter('MLfeatures.xlsx') as writer:
        corr_matrix.to_excel(writer, sheet_name='corr_matrix_spearman')
        df2.to_excel(writer, sheet_name='Sheet_name_2')