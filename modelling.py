#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:50:36 2018

@author: Sinnik
"""
import pickle

with open("./clean_train.pkl", 'rb') as f:
    clean_train_df = pickle.load(f)
