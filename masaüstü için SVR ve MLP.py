def set_data():
    import pandas as pd
    import numpy as np
    data = pd.read_csv("movie_revenue_data.csv")
    data = pd.DataFrame(data)

    years = (data["Year"]).unique()
    years.sort()
    revenues = []
    for year in (data["Year"]).unique():
        revenues.append(data[data["Year"] == year]["WorldwideBox Office"].mean())

    scaled_revenues = np.array(revenues)/1e8
    year_revenue_dict = {years[i]: scaled_revenues[i] for i in range(len(years))}
    data['Year'] = data['Year'].map(year_revenue_dict)
    data["Rating"] = data["Rating"]/10
    data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'WorldwideBox Office']] = np.log2(data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'WorldwideBox Office']])
    data = data[data["WorldwideBox Office"]>13]

    import math 
    SEED = int(math.sqrt(201401004 + 191401009))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['Title','WorldwideBox Office'], axis=1), data['WorldwideBox Office'], test_size=0.10, random_state=SEED)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1111111111111111, random_state=SEED)

    from pickle import dump
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]= min_max_scaler.fit_transform(X_train[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])
    X_validation[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]= min_max_scaler.transform(X_validation[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])
    X_test[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]= min_max_scaler.transform(X_test[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])
    dump(min_max_scaler, open('MinMaxScaler.pickle', 'wb'))

    return data, X_train, X_validation, X_test, y_train, y_validation, y_test

import math 
SEED = int(math.sqrt(201401004 + 191401009))
data, X_train, X_validation, X_test, y_train, y_validation, y_test = set_data()



import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
grid = {
    'C': [0.1, 1, 10, 100],
    "kernel": ['poly', 'rbf'],
    "gamma": [0.1, 1, 5, 10],
    "degree": [2, 3, 4],
    "max_iter": [4000, 5000, 6000, -1]
    }
SVR = SVR()
SVR_grid = GridSearchCV(SVR, grid, refit = True, n_jobs=-1, cv=10)
SVR_grid.fit(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ", SVR_grid.best_params_)
print("Support Vector Machine score on train:", SVR_grid.best_score_)
SVR_best = SVR_grid.best_estimator_

import pickle
pickle.dump(SVR_grid, open('SVR_grid.pickle', 'wb'))
    
    
    
    

import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

grid = {
        "hidden_layer_sizes": [(50), (100), (50, 50), (100, 100)],
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "momentum": [0.6, 0.9],
        "batch_size": [200, 300, 400],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "learning_rate_init": [0.001, 0.005, 0.01],
        "power_t": [2/3, 1/2, 1/3],
        "max_iter": [200, 300, 400],
        }

MLP = MLPClassifier(random_state=SEED, shuffle=True, verbose=1, early_stopping=True)

MLP_grid = GridSearchCV(MLP, grid, refit = True, n_jobs=-1, cv=10)
MLP_grid.fit(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ", MLP_grid.best_params_)
print("RandomForestRegressor score on train:", MLP_grid.best_score_)
MLP_best = MLP_grid.best_estimator_

import pickle
pickle.dump(MLP_grid, open('MLP_grid.pickle', 'wb'))
