
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split, KFold
from surprise import accuracy
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline
import pandas as pd


# Ratings data
#HW5
#Read data from “ratings small.csv” with line format: 'userID movieID ratingtimestamp'
file_path = pd.read_csv('/Users/renitawashburn/Downloads/ratings_small.csv')
#need to instantiate a “reader” object and indicate the rating scale of your ratings data
#reader = Reader(rating_scale = (0.5, 5))
reader = Reader(
    line_format='userId movieId rating', rating_scale = (0.5,5)
    )

# The columns must correspond to user id, item id and ratings (in that order).
#data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
data = Dataset.load_from_file(file_path, reader=reader)

#Spit into train and test datasets
#testset contains randomly selected user / item ratings, as opposed to full users / items
trainset, testset = train_test_split(data, test_size=.2)

#Compute the average MAE and RMSE of the Probabilistic Matrix Factorization
#(PMF), User based Collaborative Filtering, Item based Collaborative Filtering,
# Run 5-fold cross-validation and print results.
##FIT MODEL: ITEM BASED
my_k = 15
my_min_k = 5
my_sim_option = {
    'name':'pearson', 'user_based':False,
    }
algo = KNNWithMeans(k = my_k, min_k = my_min_k, sim_option = my_sim_option)
algo.fit(trainset)

results = cross_validate(algo = algo, data = data, measures=['RMSE'],
    cv=5, return_train_measures=True)
print(results['test_rmse'].mean())
#Cross validate ITEM BASED
results2 = cross_validate(algo = algo, data = data, measures=['MAE'],
    cv=5, return_train_measures=True)
print(results2['test_mae'].mean())

#Compare the average (mean) performances of User-based collaborative filtering,
#item-based collaborative filtering, PMF with respect to RMSE and MAE.

predictions = algo.test(testset)
print(accuracy.rmse(predictions))
print(accuracy.mae(predictions))

##FIT MODEL: USER BASED
my_k = 15
my_min_k = 5
my_sim_optionu = {
    'name':'pearson', 'user_based':True,
    }
algou = KNNWithMeans(k = my_k, min_k = my_min_k, sim_option = my_sim_optionu)
algou.fit(trainset)

resultsu = cross_validate(algo = algou, data = data, measures=['RMSE'],
    cv=5, return_train_measures=True)
print(resultsu['test_rmse'].mean())
#Cross validate ITEM BASED
results2u = cross_validate(algo = algou, data = data, measures=['MAE'],
    cv=5, return_train_measures=True)
print(results2u['test_mae'].mean())

#Compare the average (mean) performances of User-based collaborative filtering,
#item-based collaborative filtering, PMF with respect to RMSE and MAE.

predictionsu = algou.test(testset)
print(accuracy.rmse(predictionsu))
print(accuracy.mae(predictionsu))

#PMF: SVD Approach
algos = SVD()

# Run 5-fold cross-validation and print results
cross_validate(algos, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

predictionsv = algos.test(testset)
print(accuracy.rmse(predictionsv))
print(accuracy.mae(predictionsv))

#Examine how the cosine, MSD (Mean Squared Difference), and Pearson
#similarities impact the performances of User based Collaborative Filtering and
#Item based Collaborative Filtering. Plot your results.
sim_optionsc = {
    "name": "cosine",
    "user_based": False,  # Compute  similarities between items
}

sim_optionsm = {
    "name": "MSD",
    "user_based": False,  # Compute  similarities between items
}

sim_optionsp = {'name': 'pearson_baseline',
               'user_based': False,  # Compute  similarities between items
               }
algoc = KNNWithMeans(k = my_k, min_k = my_min_k, sim_option = sim_optionsc)
algom = KNNWithMeans(k = my_k, min_k = my_min_k, sim_option = sim_optionsm)
algop = KNNWithMeans(k = my_k, min_k = my_min_k, sim_option = sim_optionsp)

algoc.fit(trainset)
algom.fit(trainset)
algop.fit(trainset)

predictionsc = algoc.test(testset)
print(accuracy.rmse(predictionsc))
predictionsm = algom.test(testset)
print(accuracy.rmse(predictionsm))
predictionsp = algop.test(testset)
print(accuracy.rmse(predictionsp))

c= accuracy.rmse(predictionsc)
m=accuracy.rmse(predictionsm)
p=accuracy.rmse(predictionsp)
#print(algoc, algom, algop)

##USER BASED
sim_optionsc2 = {
    "name": "cosine",
    "user_based": True,  # Compute  similarities between items
}

sim_optionsm2 = {
    "name": "MSD",
    "user_based": True,  # Compute  similarities between items
}

sim_optionsp2 = {'name': 'pearson_baseline',
               'user_based': True,  # Compute  similarities between items
               }
algoc2 = KNNWithMeans(k = my_k, min_k = my_min_k, sim_option = sim_optionsc2)
algom2 = KNNWithMeans(k = my_k, min_k = my_min_k, sim_option = sim_optionsm2)
algop2 = KNNWithMeans(k = my_k, min_k = my_min_k, sim_option = sim_optionsp2)

algoc2.fit(trainset)
algom2.fit(trainset)
algop2.fit(trainset)

predictionsc2 = algoc2.test(testset)
print(accuracy.rmse(predictionsc2))
predictionsm2 = algom2.test(testset)
print(accuracy.rmse(predictionsm2))
predictionsp2 = algop2.test(testset)
print(accuracy.rmse(predictionsp2))

c2= accuracy.rmse(predictionsc2)
m2=accuracy.rmse(predictionsm2)
p2=accuracy.rmse(predictionsp2)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
meas = ['ItemCosine', 'ItemMSD', 'ItemPearson', 'UserCosine', 'UserMSD','UserPearson']
score = [c,m,p,c2,m2,p2]
# Horizontal Bar Plot
plt.bar(meas, score)
plt.show()



#Examine how the number of neighbors impacts the performances of User based
#Collaborative Filtering and Item based Collaborative Filtering? Plot your results.

kf = KFold(n_splits=3)
algo = KNNBasic(k=20, min_k=1)
best_algo = None
best_rmse = 1000.0
best_pred = None
for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)
    if rmse < best_rmse:
        best_algo = algo
        best_pred = predictions

predictionsk3 = algo.test(testset)
k3= accuracy.rmse(predictionsc)

kf2 = KFold(n_splits=3)
algo5 = KNNBasic(k=5, min_k=1)
best_algo = None
best_rmse = 1000.0
best_pred = None
for trainset, testset in kf2.split(data):
    # train and test algorithm.
    algo5.fit(trainset)
    predictions5 = algo5.test(testset)
    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)
    if rmse < best_rmse:
        best_algo = algo5
        best_pred = predictions5

predictionsk5 = algo5.test(testset)
k5= accuracy.rmse(predictions5)

fig = plt.figure(figsize=(10, 7))
ks = ['K20', 'K5']
scores = [k3,k5]
# Horizontal Bar Plot
plt.bar(ks, scores)
plt.show()

#Is the best K of User based collaborative filtering the same with the best K of Item based collaborative filtering

sim_optionsuser = {
    "name": "cosine",
    "user_based": True,  # Compute  similarities between users
}

sim_optionsitem = {
    "name": "cosine",
    "user_based": False,  # Compute  similarities between items
}

kf = KFold(n_splits=3)
algo = KNNBasic(k=20, min_k=1, sim_option=sim_optionsitem)
best_algo = None
best_rmse = 1000.0
best_pred = None
for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)
    if rmse < best_rmse:
        best_algo = algo
        best_pred = predictions

predictionsk3 = algo.test(testset)
k3= accuracy.rmse(predictionsc)

kf2 = KFold(n_splits=3)
algo5 = KNNBasic(k=5, min_k=1, sim_option=sim_optionsitem)
best_algo = None
best_rmse = 1000.0
best_pred = None
for trainset, testset in kf2.split(data):
    # train and test algorithm.
    algo5.fit(trainset)
    predictions5 = algo5.test(testset)
    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)
    if rmse < best_rmse:
        best_algo = algo5
        best_pred = predictions5

predictionsk5 = algo5.test(testset)
k5= accuracy.rmse(predictions5)

fig = plt.figure(figsize=(10, 7))
ks = ['K20', 'K5']
scores = [k3,k5]
# Horizontal Bar Plot
plt.bar(ks, scores)
plt.show()