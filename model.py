#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:16:37 2018

@author: Sinnik
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

class Model:
    
    def __init__(self, train_data_path, test_data_path, add_extra_feat):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.add_extra_feat = add_extra_feat
    
    def open_file(self):
        with open(train_data_path, 'rb') as f:
            self.train_df = pickle.load(f)
        with open(test_data_path, 'rb') as f:
            self.test_df = pickle.load(f)
        return self.train_df, self.test_df
    
    def visualise_data(self):
        self.train_df, self.test_df = self.open_file()
        self.train_df.info()
        self.train_df.describe()

        self.train_df.hist(bins = 50)
        plt.show()
        
        plt.scatter(self.train_df['temperature'], self.train_df['load'], alpha = 0.5)
        plt.xlabel("Temperature (C)")
        plt.ylabel("load (kWh)")
        corr_matrix = self.train_df.corr()
        print("Correlation Matrix")
        print(corr_matrix)
        
    def check_for_outliers(self):
        self.train_df, self.test_df = self.open_file()
        # calculate summary statistics
        data_mean, data_std = np.mean(self.train_df["temperature"]), np.std(self.train_df["temperature"])
        # As we see, the temperature follows the normal distribution. Hence the following process for
        # detection of outliers
        # identify outliers
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off

        print(self.train_df[np.logical_or(self.train_df["temperature"] > upper, 
                                          self.train_df["temperature"] < lower)])

        # By inspecting the data, we see that the very hot temperatures were either in the summer
        # or in September 2016 which was a historical high for the country. Therefore, they shouldn't
        # be discarded. 
        
        # The load distribution is bi-modal. Hence it is either non Gaussian, 
        # or a combination of two Gaussians. The latter case would be an interesting case 
        # to investigate whether we could split into two separate cases according to the values
        # of a categorical variable
        # The IQR method will be used for outlier detection

        q25, q75 = np.percentile(self.train_df["load"], 25), np.percentile(self.train_df["load"], 75)
        iqr = q75 - q25

        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off

        print(self.train_df[np.logical_or(self.train_df["load"] > upper, self.train_df["load"] < lower)])
        # This method yields no outliers. As a first approach, we are keeping all values
        
    def clean_data(self):
        self.train_df, self.test_df = self.open_file()
        # As it is concluded by calling the above method, no outliers will be discarded as a 
        # first approach
        # From the output of the info method, we can see that there are 1398 NaN values in the load
        # column. It is chosen to remove these values. Another possibility would be to interpolate
        # or exploit the seasonality of the time-series, (but as a first approach and due to the fact
        # that there are not enough data (spanning through more years for example), it is chosen to
        # omit the NaN's)
        self.clean_train_df = self.train_df[self.train_df["temperature"].notna()]
        self.clean_test_df = self.test_df[self.test_df["temperature"].notna()]
        return self.clean_train_df, self.clean_test_df
    
    def add_features(self):
        self.clean_train_df, self.clean_test_df = self.clean_data()
        # Feature 1 day of the week.
        
        # This feature is chosen, because consumers would be expected to behave 
        # differently in the weekends from the weekdays
        self.clean_train_df['day_of_week'] = \
        self.clean_train_df.index.dayofweek.astype('category', copy = False)
        
        # Feature 2 time of the day
        
        # It is assumed that gas consumption follows a pattern, in which
        # from 00:00 to 05:00 the load is small, because people tend to be inactive,
        # from 06:00 to 07:00 there is a morning ramp, from 08:00 to 19:00 working hours,
        # αnd from 20-23 nighttime
        
        hourly_index = self.clean_train_df.index.hour
        conditions = [(hourly_index >= 0) & (hourly_index <= 5), 
                      (hourly_index >= 6) & (hourly_index <= 7),
                      (hourly_index >= 8) & (hourly_index <= 19),
                      (hourly_index >= 20) & (hourly_index <= 23)]
        choices = ['early_morning', 'morning_ramp', 'working_hours', 'night_time']
        self.clean_train_df['time_of_day'] = np.select(conditions, choices)
        # Feature 3 season of the year: Since the temperature tends to be affected by the seasons,
        # adding the season as a categorical variable could improve the model performance
        monthly_index = self.clean_train_df.index.month
        conditions = [(monthly_index >= 6) & (monthly_index <= 8),
                      (monthly_index >= 9) & (monthly_index<= 11),
                      (monthly_index == 12) | (monthly_index <= 2),
                      (monthly_index >= 3) & (monthly_index <= 5)]
        choices = ['summer', 'autumn', 'winter', 'spring']
        self.clean_train_df['season'] = np.select(conditions, choices)
        self.clean_train_df = pd.get_dummies(self.clean_train_df)
        self.clean_train_df = self.clean_train_df.rename(columns = 
                                                        {'day_of_week_0': 'Sunday', 
                                                         'day_of_week_1': 'Monday', 
                                                         'day_of_week_2': 'Tuesday',
                                                         'day_of_week_3': 'Wednesday',
                                                         'day_of_week_4': 'Thursday',
                                                         'day_of_week_5': 'Friday',
                                                         'day_of_week_6': 'Saturday',
                                                         'time_of_day_early_morning': 'early_morning', 
                                                         'time_of_day_morning_ramp': 'morning_ramp', 
                                                         'time_of_day_working_hours': 'working_hours', 
                                                         'time_of_day_night_time': 'night_time',
                                                         'season_autumn': 'autumn', 
                                                         'season_winter': 'winter', 
                                                         'season_spring': 'spring', 
                                                         'season_summer': 'summer'})
        
        return self.clean_train_df
        
    def fit(self):
        self.clean_train_df = self.add_features()
        lin_reg = LinearRegression()
        lin_reg.fit(self.clean_train_df[["temperature"]], 
            self.clean_train_df[["load"]])
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(self.clean_train_df[["temperature"]], 
            self.clean_train_df[["load"]])
        forest_reg = RandomForestRegressor()
        forest_reg.fit(self.clean_train_df[["temperature"]], 
            self.clean_train_df[["load"]])
        
       
    def predict(self):
        pass
    def score(self):
        pass
    def report(self):
        # Compare the three methods and print a small message on which one performs better
        pass

'''def main():
    add_extra_feat = 'start'
    while (add_extra_feat != 'Yes' and add_extra_feat != 'No'):
        add_extra_feat = input('Do you want to add extra features? (Yes or No) ')
    if add_extra_feat == 'Yes':
        add_extra_feat = True
        which_feat = [input('Which features do you want to add? ')]
    model = Model("./train.pkl", "./test.pkl", add_extra_feat, which_feat)
    model.clean_data()
    model.fit()'''
