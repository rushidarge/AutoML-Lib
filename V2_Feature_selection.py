import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectPercentile

class Variables():
    def __init__(self,X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def variance(self):
        var_thres=VarianceThreshold(threshold=0)
        var_thres.fit(self.X_test)
        constant_columns = [column for column in X.columns
                    if column not in X.columns[var_thres.get_support()]]
        print(len(constant_columns))
        for feature in constant_columns:
            print(feature)
        # return self.X_train.columns[var_thres.get_support()], self.X_train.drop(constant_columns,axis=1)
        return self.X_train.drop(constant_columns,axis=1)

    def correlation(self, threshold=0.9):
        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = self.X_train.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        # self.X_train.drop(col_corr,axis=1)
        # self.X_test.drop(col_corr,axis=1)
        return self.X_train.drop(col_corr,axis=1), self.X_test.drop(col_corr,axis=1)

    def info_gain(self):
        mutual_info = mutual_info_classif(X_train, y_train)
        mutual_info = pd.Series(mutual_info)
        mutual_info.index = X_train.columns
        mutual_info.sort_values(ascending=False)
        print(mutual_info)
