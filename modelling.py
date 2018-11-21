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
standardized_clean_train_df = pd.DataFrame(standardized_clean_train_df, 
                                           columns = clean_train_df.columns)
lin_reg = LinearRegression()
lin_reg.fit(self.clean_train_df.drop("load", axis=1), 
            self.clean_train_df[["load"]])

scores_lin = cross_val_score(lin_reg, 
                             standardized_clean_train_df[["temperature"]],
                             standardized_clean_train_df[["load"]], 
                             scoring = "r2", cv = 10)

print(np.mean(scores_lin))

tree_reg = DecisionTreeRegressor()

scores_tree = cross_val_score(tree_reg, 
                              standardized_clean_train_df[["temperature"]],
                              standardized_clean_train_df[["load"]], 
                              scoring = "r2", cv = 10)

print(np.mean(scores_tree))

forest_reg = RandomForestRegressor()

scores_forest = cross_val_score(forest_reg, 
                                standardized_clean_train_df[["temperature"]],
                                standardized_clean_train_df[["load"]], 
                                scoring = "r2", cv = 10)

print(np.mean(scores_forest))

param_grid = [{'n_estimators': [40], 'max_features':[13]}]
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = 'r2')
grid_search.fit(test.drop("load", axis=1), test[["load"]])
grid_search.best_params_
grid_search.cv_results_
feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, list(test.drop("load", axis=1).columns)),reverse = True)
forest_reg = RandomForestRegressor(n_estimators= 40)
scores_forest = cross_val_score(forest_reg, test[['temperature','Sunday', 'Monday', 'Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday', 'summer']],test[["load"]], scoring = "r2", cv = 10)

print(np.mean(scores_forest))


scores_forest = cross_val_score(forest_reg, standardized_clean_train_df[["temperature"]],standardized_clean_train_df[["load"]], scoring = "r2", cv = 10)

print(np.mean(scores_forest))

poly = PolynomialFeatures(2)
pol_features = poly.fit_transform(clean_train_df[['temperature']])

pol_features_df = pd.DataFrame(pol_features[:,2], columns = ["temperature_squared"])
pol_features_df.index = clean_train_df.index

clean_train_poly_df = clean_train_df.join(pol_features_df)

lin_reg = LinearRegression()
scores_lin = cross_val_score(lin_reg, clean_train_poly_df[["temperature", 
                                                           "temperature_squared"]],
                                      clean_train_poly_df[["load"]], 
                                      scoring = "r2", cv = 10)
print(np.mean(scores_lin))

scores_tree = cross_val_score(tree_reg, clean_train_poly_df[["temperature", 
                                                             "temperature_squared"]],
                                        clean_train_poly_df[["load"]], 
                                        scoring = "r2", cv = 10)

print(np.mean(scores_tree))

scores_forest = cross_val_score(forest_reg, clean_train_poly_df[["temperature", 
                                                                 "temperature_squared"]],
                                            clean_train_poly_df[["load"]], 
                                            scoring = "r2", cv = 10)

print(np.mean(scores_forest))

# Using the squared temperature really didn't help. Will look for other features

####### Try to re-do the feature engineering with sklearn's OneHotEncoder
# Feature 1 day of the week
clean_train_df['day_of_week'] = clean_train_df.index.dayofweek.astype('category', copy = False)
clean_train_df = pd.get_dummies(clean_train_df)

'''clean_train_df = clean_train_df.drop(['day_of_week_0', 
                                      'day_of_week_1', 
                                      'day_of_week_2', 
                                      'day_of_week_3', 
                                      'day_of_week_4', 
                                      'day_of_week_5', 
                                      'day_of_week_6'], 
                                       axis = 1)'''
scores_lin = cross_val_score(lin_reg, clean_train_df[["temperature",'day_of_week_0', 
                                      'day_of_week_1', 
                                      'day_of_week_2', 
                                      'day_of_week_3', 
                                      'day_of_week_4', 
                                      'day_of_week_5', 
                                      'day_of_week_6']],
                             clean_train_df[["load"]], 
                             scoring = "r2", cv = 10)
print(np.mean(scores_lin))

scores_tree = cross_val_score(tree_reg, clean_train_df[["temperature", 'day_of_week_0', 
                                      'day_of_week_1', 
                                      'day_of_week_2', 
                                      'day_of_week_3', 
                                      'day_of_week_4', 
                                      'day_of_week_5', 
                                      'day_of_week_6']],
                              clean_train_df[["load"]], 
                              scoring = "r2", cv = 10)
print(np.mean(scores_tree))

scores_forest = cross_val_score(forest_reg, clean_train_df[["temperature", 'day_of_week_0', 
                                      'day_of_week_1', 
                                      'day_of_week_2', 
                                      'day_of_week_3', 
                                      'day_of_week_4', 
                                      'day_of_week_5', 
                                      'day_of_week_6']],
                                clean_train_df[["load"]], 
                                scoring = "r2", cv = 10)
print(np.mean(scores_forest))

# Feature 2 time of the day
# It is assumed that gas and electricity consumption follow a similar pattern
# So, from 0-5 early morning, 6-7 morning ramp, 8-19 working hours, 20-23 nighttime
clean_train_df['time'] = clean_train_df.index.hour
conditions = [(clean_train_df['time'] >= 0) & (clean_train_df['time'] <= 5), (clean_train_df['time'] >= 6) & (clean_train_df['time'] <= 7),(clean_train_df['time'] >= 8) & (clean_train_df['time'] <= 19),(clean_train_df['time'] >= 20) & (clean_train_df['time'] <= 23)]
choices = ['early_morning', 'morning_ramp', 'working_hours', 'night_time']
clean_train_df['time_of_day'] = np.select(conditions, choices)
clean_train_df = pd.get_dummies(clean_train_df)

clean_train_df = clean_train_df.rename(columns = {'time_of_day_early_morning': 'early_morning', 
                                                  'time_of_day_morning_ramp': 'morning_ramp', 
                                                  'time_of_day_working_hours': 'working_hours', 
                                                  'time_of_day_night_time': 'night_time'})
    
scores_lin = cross_val_score(lin_reg, clean_train_df[['temperature','day_of_week', 
                                                      'early_morning', 'morning_ramp',
                                                      'working_hours','night_time']],
                                      clean_train_df[["load"]], scoring = "r2", cv = 10)
print(np.mean(scores_lin))

scores_tree = cross_val_score(tree_reg, clean_train_df[['temperature','day_of_week',
                                                        'early_morning', 'morning_ramp',
                                                        'working_hours','night_time']],
                                        clean_train_df[["load"]], scoring = "r2", cv = 10)
print(np.mean(scores_tree))

scores_forest = cross_val_score(forest_reg, clean_train_df[['temperature', 'day_of_week',
                                                            'early_morning', 'morning_ramp',
                                                            'working_hours','night_time']],
                                            clean_train_df[["load"]], scoring = "r2", cv = 10)
print(np.mean(scores_forest))
# Feature 3 season of the year 
# 6-8 Summer 
# 9-11 Autumn
# 12-2 Winter
# 3-5 Spring
# Try https://stackoverflow.com/questions/26886653/pandas-create-new-column-based-on-values-from-other-columns
conditions = [(clean_train_df.index.month >= 6) & (clean_train_df.index.month <= 8),(clean_train_df.index.month >= 9) & (clean_train_df.index.month <= 11),(clean_train_df.index.month == 12) | (clean_train_df.index.month <= 2),(clean_train_df.index.month >= 3) & (clean_train_df.index.month <= 5)]

choices = ['summer', 'autumn', 'winter', 'spring']
clean_train_df['season'] = np.select(conditions, choices)
clean_train_df = pd.get_dummies(clean_train_df)
clean_train_df = clean_train_df.rename(columns = {'season_autumn': 'autumn', 
                                                  'season_winter': 'winter', 
                                                  'season_spring': 'spring', 
                                                  'season_summer': 'summer'})
scores_lin = cross_val_score(lin_reg, clean_train_df[['temperature','day_of_week', 
                                                      'autumn', 'winter',
                                                      'spring','summer']],
                                      clean_train_df[["load"]], scoring = "r2", cv = 10)
print(np.mean(scores_lin))

scores_tree = cross_val_score(tree_reg, clean_train_df[['temperature','day_of_week', 
                                                      'autumn', 'winter',
                                                      'spring','summer']],
                              clean_train_df[["load"]], 
                              scoring = "r2", cv = 10)
print(np.mean(scores_tree))

scores_forest = cross_val_score(forest_reg, clean_train_df[['temperature','day_of_week', 
                                                      'autumn', 'winter',
                                                      'spring','summer']],
                                            clean_train_df[["load"]], scoring = "r2", cv = 10)
print(np.mean(scores_forest))


with open("./test.pkl", 'rb') as f:
    test_df = pickle.load(f)

test_df.info()
clean_test_df = test_df[test_df['temperature'].notna()]
clean_test_df['day_of_week'] = clean_test_df.index.dayofweek.astype('category', copy = False)
tree_reg.fit(clean_train_df[["temperature", "day_of_week"]],
                              clean_train_df[["load"]])
tree_reg.predict(clean_test_df)
