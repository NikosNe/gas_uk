# -*- coding: utf-8 -*-
import pandas as pd
import pickle 
import matplotlib.pyplot as plt 
import numpy as np
%matplotlib qt

with open("./clean_train.pkl", 'rb') as f:
    train_df = pickle.load(f)

train_df.info()
train_df.describe()

train_df.hist(bins = 50)
plt.show()

# calculate summary statistics
data_mean, data_std = np.mean(train_df["temperature"]), np.std(train_df["temperature"])
# As we see, the temperature follows the normal distribution. Hence the following process for
# detection of outliers
# identify outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off

print(train_df[np.logical_or(train_df["temperature"] > upper, train_df["temperature"] < lower)])

# By inspecting the data, we see that the very hot temperatures were either in the summer
# or in September 2016 which was a historical high for the country. Therefore, they shouldn't
# be discarded. 

# The load distribution is bi-modal. Hence it is either non Gaussian, or a combination of two Gaussians
# The IQR method will be used for outlier detection

q25, q75 = np.percentile(train_df["load"], 25), np.percentile(train_df["load"], 75)
iqr = q75 - q25

cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off

print(train_df[np.logical_or(train_df["load"] > upper, train_df["load"] < lower)])

# No outliers

# From the output of the info method, we can see that there are 1398 NaN values in the load
# column. It is chosen to remove these values. Another possibility would be to interpolate
# or exploit the seasonality of the time-series, (((but as a first approach and due to the fact
# that there are not enough data (spanning through more years for example), it is chosen to
# omit the NaN's)))

clean_train_df = train_df[train_df["temperature"].notna()]
clean_train_df.to_pickle("./clean_train.pkl")

# Check for correlation between the two variables

from pandas.tools.plotting import scatter_matrix
scatter_matrix(clean_train_df)
corr_matrix = clean_train_df.corr()
print(corr_matrix)

# Load and temperature are very strongly negatively correlated. 
# The correlation coefficient is very high (-0.938645), which suggests that linear models
# could work for this case
