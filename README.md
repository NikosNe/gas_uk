# Coding Exercise

## Introduction

The following exercise will try to quickly assess your timeseries modelling and data science skills.
You are provided with two pickle files, *train.pkl* and *test.pkl*. The names should be pretty self-explanatory.

*train.pkl* is a time-indexed pandas dataframe with two columns, "temperature" and "load". Units are degrees C and kWh respectively. Column "load" is a timeseries representing consumption of gas in a specific location in UK. Column "temperature" contains temperature measurements for the same location. *test.pkl* only contains the temperature column. 

## Task

Your goal is to create an installable python package to forecast the load as a function of the datetime and temperature. 
The package should contain a class called *Model* with an sklearn-style API (at least *fit*, *predict* and *score* methods).
By installable we mean that we should be able to install your package by running e.g. ```pip install -e git+https://github.com/user/repo.git@branch```

There are no limitations in terms of modelling techniques or extra data you might want to include. Feel free to do any type of feature engineering or apply any algorithm as long as your results are reproducible. If you do incorporate extra data, please include the code for that in the package even if the extra features end up giving no value. Similarly, if you try several estimators, please do include them in the class. We will discuss the feature engineering and model creation process at the interview. 

You are welcome to use part of the training set to tune your model.
If the training of the model takes more than 20 minutes on a normal laptop please make sure to include a method in the class to quickly load the trained model (e.g. by pickling it and adding it to the package).

You will be evaluated on:
- feature engineering and model creation
- code quality 
- the score of the model on the test set (sklearn r2_score)

##  Delivering

You are welcome to reply to the email you will get from Orsted HR with a link to a GitHub or GitLab repo containing your code, Cc edsim@orsted.dk. Please include any dependencies in the *requirements.txt* file of your package. Please do not push to this repository, unless you want the other candidates to be able to look at your code!
The whole exercise should not take you a lot more than two to three hours. 
