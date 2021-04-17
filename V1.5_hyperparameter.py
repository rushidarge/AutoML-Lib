from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time

class Parameters():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = {}
        self.model['knnclf'] = KNeighborsClassifier()
        self.model['lrclf'] = LogisticRegression()
        self.model['gnbclf'] = GaussianNB()
        self.model['svcclf'] = SVC()
        self.model['sgdclf'] = SGDClassifier()
        self.model['dtclf'] = DecisionTreeClassifier()
        self.model['rfclf'] = RandomForestClassifier()

        self.cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        self.parameters = {}
        self.parameters['gbclf'] = {}
        self.parameters['gbclf']['n_estimators'] = [10, 100, 1000]
        self.parameters['gbclf']['learning_rate'] = [0.001, 0.01, 0.1]
        self.parameters['gbclf']['subsample'] = [0.5, 0.7, 1.0]
        self.parameters['gbclf']['max_depth'] = [3, 7, 9]
        self.parameters['gbclf']['max_features'] = ['sqrt', 'log2']

        self.parameters['rfclf'] = {}
        self.parameters['rfclf']['n_estimators'] = [10, 100, 1000]
        self.parameters['rfclf']['criterion'] = ['gini', 'entropy']
        self.parameters['rfclf']['max_features'] = ['sqrt', 'log2']

        self.parameters['svcclf'] = {}
        self.parameters['svcclf']['kernel'] = ['poly', 'rbf', 'sigmoid']
        self.parameters['svcclf']['C'] = [50, 10, 1.0, 0.1, 0.01]
        self.parameters['svcclf']['gamma'] = ['scale']

        self.parameters['knnclf'] = {}
        self.parameters['knnclf']['n_neighbors'] = range(1, 21, 2)
        self.parameters['knnclf']['weights'] = ['uniform', 'distance']
        self.parameters['knnclf']['metric'] = ['euclidean', 'manhattan', 'minkowski']

        self.parameters['lrclf'] = {}
        self.parameters['lrclf']['solvers'] = ['newton-cg', 'lbfgs', 'liblinear']
        self.parameters['lrclf']['penalty'] = ['l1', 'l2', 'elasticnet']
        self.parameters['lrclf']['c_values'] = [100, 10, 1.0, 0.1, 0.01]

        self.parameters['sgdclf'] = {}
        self.parameters['sgdclf']['n_iter'] = [1, 5, 10]
        self.parameters['sgdclf']['alpha'] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        self.parameters['sgdclf']['penalty'] = ["none", "l1", "l2"]

        self.parameters['gnbclf'] = {}
        self.parameters['gnbclf']['var_smoothing'] = np.logspace(0,-9, num=100)

        self.parameters['dtclf'] = {}
        self.parameters['dtclf']['min_samples_split'] = range(1,10)
        self.parameters['dtclf']['min_samples_leaf'] = range(1,5)
        self.parameters['dtclf']['max_depth'] = range(1,10)
        self.parameters['dtclf']['criterion'] = ['gini', 'entropy']

    def hypertune(self,model_name):
        start = time.time()
        self.model[model_name].fit(self.X_train, self.y_train)
        end = time.time()
        model_time = end-start

        sum = 1
        for model, para in self.parameters[model_name].items():
            sum = sum * len(para)
        print('Estimate time = ',np.round((sum*model_time*3)/60,3))
        
        grid_search = GridSearchCV(estimator=self.model[model_name], param_grid=self.parameters[model_name], n_jobs=-1, cv=self.cv, scoring='accuracy',error_score=0)
        grid_result = grid_search.fit(X_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        return grid_result
