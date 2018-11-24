
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
    def __init__(self, train_data_path, test_data_path, 
                 rand_for_filename, impute_or_remove):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.rand_for_filename = rand_for_filename
        self.impute_or_remove = impute_or_remove
        
    def open_file(self, path):
        '''This method receives a file path of a pickle and 
           returns its output'''
        with open(path, 'rb') as f:
            self.df = pickle.load(f)
        return self.df
    
    def visualise_data(self):
        '''This method makes the needed visualisations as part of the EDA'''
        self.train_df = self.open_file(self.train_data_path)
        self.train_df.info()
        self.train_df.describe()

        self.train_df.hist(bins=50)
        plt.show()

        plt.scatter(self.train_df['temperature'],
                    self.train_df['load'], alpha= 0.5)
        plt.xlabel("Temperature (C)")
        plt.ylabel("load (kWh)")
        plt.show()
        plt.scatter(self.train_df.index,
                    self.train_df['load'], alpha = 0.5)
        
        plt.xlabel("datetime")
        plt.ylabel("load (kWh)")
        plt.show()
        corr_matrix = self.train_df.corr()
        print("Correlation Matrix")
        print(corr_matrix)

    def check_for_outliers(self):
        '''This method checks whether there are outliers in the given 
        variables. As it is, it doesn't return anything, but it is left
        for reference purposes'''
        self.train_df = self.open_file(self.train_data_path)
        # calculate summary statistics
        data_mean, data_std = (np.mean(self.train_df["temperature"]),
                               np.std(self.train_df["temperature"]))
        # As we see, the temperature follows the normal distribution.
        # Hence the following process for detection of outliers
        # identify outliers
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off
        print(self.train_df[np.logical_or(
              self.train_df["temperature"] > upper,
              self.train_df["temperature"] < lower)])

        # By inspecting the data, we see that the very hot temperatures were
        # either in the summer or in September 2016 which was a historical high
        # for the country. Therefore, they shouldn't be discarded.

        # The load distribution is bi-modal. Hence it is either non Gaussian,
        # or a combination of two Gaussians. The latter case would be
        # an interesting case to investigate whether we could split into
        # two separate cases according to the values of a categorical variable
        # The IQR method will be used for outlier detection

        q25, q75 = (np.percentile(self.train_df["load"], 25),
                    np.percentile(self.train_df["load"], 75))
        iqr = q75 - q25

        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off

        print(self.train_df[np.logical_or(self.train_df["load"] > upper,
                                          self.train_df["load"] < lower)])
        # This method yields no outliers. As a first approach, we are keeping
        # all values

    def clean_data(self, path):
        '''This method receives as an argument the path of either the train.pkl
           or the test.pkl and returns them without NA's'''
           
        df = self.open_file(path)
        
        # As it is concluded by calling the above method, no outliers will be
        # discarded as a first approach.
        # From the output of the info method, we can see that there are
        # 1398 NaN values in the load column. It is chosen to remove these
        # values. Another possibility would be to interpolate or exploit
        # the periodicity of the time-series, but as a first approach and due
        # it is chosen to omit the NaN's
        
        if self.impute_or_remove == 'remove':
            df = df[df["temperature"]\
                       .notna()]
            df = df[df["temperature"].notna()]
        
        # If one wants to make a comparison with imputed values,
        # They can just comment out the following line of code and
        # comment the notna part in the lines 103-106
        else:
            df['temperature'] =\
            df['temperature'].groupby(df.index.month).transform(
                    lambda x: x.fillna(x.median()))
        return df

    def add_features(self, df):
        '''This method adds extra features in order to 
           improve the model performance. It receives as an argument
           a dataframe and returns the same argument with extra columns, one
           for each extra feature'''
        # Feature 1 day of the week.

        # This feature is chosen, because consumers would be expected to behave
        # differently in the weekends from the weekdays
        df['day_of_week'] = df.index.dayofweek.astype('category', copy=False)

        # Feature 2 time of the day

        # It is assumed that gas consumption follows a pattern, in which
        # from 00:00 to 05:00 the load is small, because people tend to be
        # inactive, from 06:00 to 07:00 there is a morning ramp,
        # from 08:00 to 19:00 working hours αnd from 20-23 nighttime.

        hourly_index = df.index.hour
        conditions = [(hourly_index >= 0) & (hourly_index <= 5),
                      (hourly_index >= 6) & (hourly_index <= 7),
                      (hourly_index >= 8) & (hourly_index <= 19),
                      (hourly_index >= 20) & (hourly_index <= 23)]
        choices = ['early_morning', 'morning_ramp',
                   'working_hours', 'night_time']
        df['time_of_day'] = np.select(conditions, choices)
        
        # Feature 3 season of the year
        
        # The gas consumption could be affected
        # by the seasons, due to holidays, etc.
        # Hence, adding the season as a categorical variable
        # could improve the model performance
        
        monthly_index = df.index.month
        conditions = [(monthly_index >= 6) & (monthly_index <= 8),
                      (monthly_index >= 9) & (monthly_index <= 11),
                      (monthly_index == 12) | (monthly_index <= 2),
                      (monthly_index >= 3) & (monthly_index <= 5)]
        choices = ['summer', 'autumn', 'winter', 'spring']
        df['season'] = np.select(conditions, choices)
        df = pd.get_dummies(df)
        
        # Renaming the dummy variables to get meaningful names
        # (Day names start with small letters for consistency)
        df = df.rename(columns={'day_of_week_0': 'sunday',
                                'day_of_week_1': 'monday',
                                'day_of_week_2': 'tuesday',
                                'day_of_week_3': 'wednesday',
                                'day_of_week_4': 'thursday',
                                'day_of_week_5': 'friday',
                                'day_of_week_6': 'saturday',
                                'time_of_day_early_morning': 'early_morning',
                                'time_of_day_morning_ramp': 'morning_ramp',
                                'time_of_day_working_hours': 'working_hours',
                                'time_of_day_night_time': 'night_time',
                                'season_autumn': 'autumn',
                                'season_winter': 'winter',
                                'season_spring': 'spring',
                                'season_summer': 'summer'})
    
        return df

    def fit(self):
        '''This method tries several models and returns their fits.
           The random forest regressor fit has been pickled and included in the
           package. To run this method just add its path as 
           the filename argument'''
          
        # Normally, I scale the data before doing the fit,
        # but in this dataset, after trying both with scaled and unscaled data,
        # the performace does not change
        self.clean_train_df = self.clean_data(self.train_data_path)
        
        self.clean_train_extra_feat_df = self.add_features(self.clean_train_df)
        self.clean_features_df = \
        self.clean_train_extra_feat_df.drop("load", axis=1)
        self.y = np.array(self.clean_train_df[["load"]]).ravel()
        # Three methods are used to solve this regression problem
        # 1. Linear Regression
        
        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.clean_features_df, self.y)
        
        # 2. Decision Trees Regression
        
        self.tree_reg = DecisionTreeRegressor()
        self.tree_reg.fit(self.clean_features_df, self.y)
        
        '''param_grid = [{'max_features':[13, 14, 15, 16]}]
        grid_search = GridSearchCV(self.tree_reg, param_grid, cv=5,
                                   scoring='r2')
        grid_search.fit(self.clean_features_df, self.y)
        print(grid_search.best_params_)
        print(grid_search.cv_results_)
        feature_importances = grid_search.best_estimator_.feature_importances_
        print(sorted(zip(feature_importances,
                     list(self.clean_features_df.columns)),
              reverse=True))'''
        
        # According to the best_params and cv_results, 13 features 
        # should be picked. And actually, the variables 'winter', 
        # "morning_ramp" and "summer" have the lowest importances. 
        # It can also be observed that temperature is far more significant
        # than all the other features, followed by the days of the week.
        # This is an indication that the times of the day should be
        # modelled differently
        
        # 3. Random Forest Regression
        # Hyperparameter tuning. This section is commented out because running 
        # it takes time. This model has been pickled
        
        '''param_grid = [{'n_estimators': [10, 40, 50, 60, 70],
                       'max_features':[13, 14, 15, 16, 17]}]
        self.forest_reg = RandomForestRegressor()
        grid_search = GridSearchCV(self.forest_reg, param_grid, cv=5,
                                   scoring='r2')
        grid_search.fit(self.clean_features_df, self.y)
        print(grid_search.best_params_)
        print(grid_search.cv_results_)
        feature_importances = grid_search.best_estimator_.feature_importances_
        print(sorted(zip(feature_importances,
                     list(self.clean_features_df.columns)),
              reverse=True))
        self.forest_reg = RandomForestRegressor(70)'''
        
        # Save model to disk
        
        # filename = 'random_forest.sav'
        # pickle.dump(self.forest_reg, open(filename, 'wb'))
        self.random_forest_model = pickle.load(open(self.rand_for_filename, 
                                                    'rb'))
        
        # According to the best_params and cv_results, 14 features and
        # 70 estimators can be picked. However, due to the nature of the
        # categorical variables, some further feature engineering should be
        # implemented, so as to e.g. split the times of the day differently
        # and therefore all features will be used as a first attempt
        
        self.random_forest_model.fit(self.clean_features_df, self.y)
        return (self.clean_features_df, self.lin_reg,
                self.tree_reg, self.random_forest_model)

    def score(self, cv_or_score, score_test_path):
        ''' This method cross validates and calculates scores.
           To do the former, give the value 'cv' to the cv_or_score
           parameter, otherwise give the value 'score'. In case
           the latter is chosen, the score_test_path should be 
           the path of the test dataframe you want to see the r2 scores
           resulting from the fit. Otherwise, the score_test_path can
           be a random string. The dataframe corresponding to the 
           score_test_path should be of the pickle format and should have 
           at least one column named "load"'''
        
        (self.clean_features_df, self.lin_reg, self.tree_reg, 
        self.random_forest_model) = self.fit()
        methods = [self.lin_reg, self.tree_reg, self.random_forest_model]
        if cv_or_score == 'cv':
            self.y = np.array(self.clean_train_df[["load"]]).ravel()
            
        elif cv_or_score == 'score':
            self.score_test_df = self.clean_data(score_test_path)
            self.clean_features_df = self.add_features(self.score_test_df)
            self.clean_features_df= self.clean_features_df.drop(["load"], 
                                                                axis=1) 
            self.y = np.array(self.score_test_df[["load"]]).ravel()
        for method in methods:
            scores = cross_val_score(method,
                                     self.clean_features_df,
                                     self.y,
                                     scoring="r2", cv=10)
            print(scores)
            print(np.mean(scores))
            
    def predict(self):
        '''This method makes the load predictions for the test file
            provided as a pickle'''
        self.clean_test_df = self.clean_data(self.test_data_path)
        (self.clean_features_df, self.lin_reg,
         self.tree_reg, self.random_forest_model) = self.fit()
        self.clean_test_extra_feat_df = self.add_features(self.clean_test_df)
        return (self.lin_reg.predict(self.clean_test_extra_feat_df),
        self.tree_reg.predict(self.clean_test_extra_feat_df),
        self.random_forest_model.predict(self.clean_test_extra_feat_df))
    
        
        
    
