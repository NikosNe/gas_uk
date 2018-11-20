#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:50:36 2018

@author: Sinnik
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

with open("./clean_train.pkl", 'rb') as f:
    clean_train_df = pickle.load(f)

# Standardize the data, so that they have the same scale
 
scaler = StandardScaler()
scaler.fit(clean_train_df)
standardized_clean_train_df = scaler.transform(clean_train_df)
standardized_clean_train_df = pd.DataFrame(standardized_clean_train_df, columns = clean_train_df.columns)
lin_reg = LinearRegression()
lin_reg.fit(standardized_clean_train_df[["temperature"]], standardized_clean_train_df[["load"]])

scores_lin = cross_val_score(lin_reg, standardized_clean_train_df[["temperature"]],standardized_clean_train_df[["load"]], scoring = "r2", cv = 10)

print(np.mean(scores_lin))

tree_reg = DecisionTreeRegressor()

scores_tree = cross_val_score(tree_reg, standardized_clean_train_df[["temperature"]],standardized_clean_train_df[["load"]], scoring = "r2", cv = 10)

print(np.mean(scores_tree))

forest_reg = RandomForestRegressor()

scores_forest = cross_val_score(forest_reg, standardized_clean_train_df[["temperature"]],standardized_clean_train_df[["load"]], scoring = "r2", cv = 10)

print(np.mean(scores_forest))

param_grid = [{'n_estimators': [40, 60, 80]}]
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = 'r2')
grid_search.fit(standardized_clean_train_df[["temperature"]],standardized_clean_train_df[["load"]])
grid_search.best_params_
grid_search.cv_results_

forest_reg = RandomForestRegressor(n_estimators = 40)

scores_forest = cross_val_score(forest_reg, standardized_clean_train_df[["temperature"]],standardized_clean_train_df[["load"]], scoring = "r2", cv = 10)

print(np.mean(scores_forest))

poly = PolynomialFeatures(2)
pol_features = poly.fit_transform(clean_train_df[['temperature']])

pol_features_df = pd.DataFrame(pol_features[:,2], columns = ["temperature_squared"])
pol_features_df.index = clean_train_df.index

clean_train_poly_df = clean_train_df.join(pol_features_df)

lin_reg = LinearRegression()
scores_lin = cross_val_score(lin_reg, clean_train_poly_df[["temperature", "temperature_squared"]],clean_train_poly_df[["load"]], scoring = "r2", cv = 10)
print(np.mean(scores_lin))

scores_tree = cross_val_score(tree_reg, clean_train_poly_df[["temperature", "temperature_squared"]],clean_train_poly_df[["load"]], scoring = "r2", cv = 10)

print(np.mean(scores_tree))

scores_forest = cross_val_score(forest_reg, clean_train_poly_df[["temperature", "temperature_squared"]],clean_train_poly_df[["load"]], scoring = "r2", cv = 10)

print(np.mean(scores_forest))

# Using the squared temperature really didn't help. Will look for other features

# Feature 1 day of the week
clean_train_df['day_of_week'] = clean_train_df.index.dayofweek.astype('category', copy = False)
clean_train_df = pd.get_dummies(clean_train_df)
clean_train_df = clean_train_df.rename(columns={'dayofweek0': 'Sunday', 'dayofweek1': 'Monday', 'dayofweek2': 'Tuesday', 'dayofweek3': 'Wednesday', 'dayofweek4': 'Thursday', 'dayofweek5': 'Friday', 'dayofweek6': 'Saturday'})
# Feature 2 time of the day

# Feature 3 season of the year