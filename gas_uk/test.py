#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import the model
from gas_uk import model
# Create an instance of the class model. As arguments, there should be given:
# The train pickle file, the test pickle file and the .sav file, all of which 
# are in the package folder. The following paths are not guaranteed to work 
# on every PC, since this depends on how the package is installed
test = model.Model("./train.pkl", "./test.pkl", './random_forest.sav')
# Make data visualisations
test.visualise_data()
# Fit three models 
test.fit()
# Check the cross validation score. If you want to check the performance 
# on another dataset, then instead of 'cv', write 'score' and instead of
# 'nothing', the path of the dataframe. Detailed instructions on the dataframe
# can be found on the comments of the code
test.score('cv', 'nothing')
# If one wants to see the score on a test score, its file path should be added
test.score('score', "./train.pkl")
# Make predictions on the given test dataset
test.predict()
