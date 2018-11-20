#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:16:37 2018

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

class Model:
    
    def __init__(self, train_data_path, test_data_path, add_extra_feat, which_feat):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.add_extra_feat = add_extra_feat
        self.which_feat = which_feat
    
    def open_file(self):
        with open(self.train_data_path, 'rb') as f:
            self.train_df = pickle.load(f)
        with open(self.test_data_path, 'rb') as f:
            self.test_df = pickle.load(f)
        return self.train_df, self.test_df
    
    def clean_data(self):
        self.train_df, self.test_df = self.open_file()
        self.clean_train_df = self.train_df[self.train_df["temperature"].notna()]
        self.clean_test_df = self.test_df[self.test_df["temperature"].notna()]
        return self.clean_train_df, self.clean_test_df
    
    def add_features(self):
        if self.add_extra_feat == False:
            pass
        else:
            if self.which_feat =='day_of_week':
                self.clean_train_df['day_of_week'] = self.clean_train_df.index.dayofweek.astype('category', copy = False)
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
        pass

def main():
    add_extra_feat = input('Do you want to add extra features? ')
    if add_extra_feat == 'Yes':
        add_extra_feat = True
    model = Model("./train.pkl", "./test.pkl", add_extra_feat, 'day_of_week')
    model.clean_data()
    model.fit()
