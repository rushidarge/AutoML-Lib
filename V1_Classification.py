# Logistic Regression Algorithm
# Naïve Bayes Algorithm
# Decision Tree Algorithm
# K-Nearest Neighbours Algorithm
# Support Vector Machine Algorithm
# Random Forest Algorithm
# Stochastic Gradient Descent Algorithm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import sklearn.metrics as sm
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import time

class Automl():
    def __init__(self,X_train, X_test, y_train, y_test, auto=True, score_type='binary'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.auto = auto
        self.acc = pd.DataFrame()
        self.score_type = score_type
        self.setting = str() 
        if self.score_type == 'binary':
            self.setting = 'binary'
        else:
            self.score_type = 'weighted'

    def knn(self):
        neigh = KNeighborsClassifier(n_neighbors=3)
        start = time.time()
        neigh.fit(self.X_train, self.y_train)
        end = time.time()
        print('KNN')
        y_pred = neigh.predict(self.X_test)
        
        data = {'Algorithm': 'KNN','Accuracy': sm.accuracy_score(self.y_test,y_pred), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type) \
                ,'Recall score': sm.recall_score(self.y_test,y_pred,average=self.score_type), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type), \
                'F1 score': sm.f1_score(self.y_test,y_pred,average=self.score_type),'time(sec)':end-start} 
        new = pd.Series(data)
        self.acc = self.acc.append(new, ignore_index=True)

    def logisticreg(self):
        lr_model = LogisticRegression()
        start = time.time()
        lr_model.fit(self.X_train, self.y_train)
        end = time.time()
        print('Logisitc Regression')
        y_pred = lr_model.predict(self.X_test)

        data = {'Algorithm': 'Logisitc Regression','Accuracy': sm.accuracy_score(self.y_test,y_pred), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type) \
                ,'Recall score': sm.recall_score(self.y_test,y_pred,average=self.score_type), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type), \
                'F1 score': sm.f1_score(self.y_test,y_pred,average=self.score_type),'time(sec)':end-start} 
        new = pd.Series(data)
        self.acc = self.acc.append(new, ignore_index=True)

    def gaussiannb(self):
        gnb = GaussianNB()
        start = time.time()
        gnb.fit(self.X_train, self.y_train)
        end = time.time()
        print('Gaussian NavieBayes')
        y_pred = gnb.predict(self.X_test)

        data = {'Algorithm': 'Gaussian NavieBayes','Accuracy': sm.accuracy_score(self.y_test,y_pred), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type) \
                ,'Recall score': sm.recall_score(self.y_test,y_pred,average=self.score_type), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type), \
                'F1 score': sm.f1_score(self.y_test,y_pred,average=self.score_type),'time(sec)':end-start} 
        new = pd.Series(data)
        self.acc = self.acc.append(new, ignore_index=True)   

    def decisiontree(self):
        clf_entropy = DecisionTreeClassifier()
        start = time.time()
        clf_entropy.fit(self.X_train, self.y_train)
        end = time.time()
        print('Decision Tree Classifier')
        y_pred = clf_entropy.predict(self.X_test)

        data = {'Algorithm': 'Decision Tree Classifier','Accuracy': sm.accuracy_score(self.y_test,y_pred), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type) \
                ,'Recall score': sm.recall_score(self.y_test,y_pred,average=self.score_type), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type), \
                'F1 score': sm.f1_score(self.y_test,y_pred,average=self.score_type),'time(sec)':end-start} 
        new = pd.Series(data)
        self.acc = self.acc.append(new, ignore_index=True)  

    def svm(self):
        svc_model = SVC()
        start = time.time()
        svc_model.fit(self.X_train, self.y_train)
        end = time.time()
        print('Support vector machine')
        y_pred = svc_model.predict(self.X_test)

        data = {'Algorithm': 'Support vector machine','Accuracy': sm.accuracy_score(self.y_test,y_pred), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type) \
                ,'Recall score': sm.recall_score(self.y_test,y_pred,average=self.score_type), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type), \
                'F1 score': sm.f1_score(self.y_test,y_pred,average=self.score_type),'time(sec)':end-start} 
        new = pd.Series(data)
        self.acc = self.acc.append(new, ignore_index=True) 

    def randomforest(self):
        Rclf = RandomForestClassifier(random_state=0)
        start = time.time()
        Rclf.fit(self.X_train, self.y_train)
        end = time.time()
        print('Random Forest Classifier')
        y_pred = Rclf.predict(self.X_test)

        data = {'Algorithm': 'Random Forest Classifier','Accuracy': sm.accuracy_score(self.y_test,y_pred), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type) \
                ,'Recall score': sm.recall_score(self.y_test,y_pred,average=self.score_type), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type), \
                'F1 score': sm.f1_score(self.y_test,y_pred,average=self.score_type),'time(sec)':end-start} 
        new = pd.Series(data)
        self.acc = self.acc.append(new, ignore_index=True)

    def sgdclassifier(self):
        sgdclf = SGDClassifier()
        start = time.time()
        sgdclf.fit(self.X_train, self.y_train)
        end = time.time()
        print('SGD Classifier')
        y_pred = sgdclf.predict(self.X_test)

        data = {'Algorithm': 'SGD Classifier','Accuracy': sm.accuracy_score(self.y_test,y_pred), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type) \
                ,'Recall score': sm.recall_score(self.y_test,y_pred,average=self.score_type), 'Precision score': sm.precision_score(self.y_test,y_pred,average=self.score_type), \
                'F1 score': sm.f1_score(self.y_test,y_pred,average=self.score_type),'time(sec)':end-start} 
        new = pd.Series(data)
        self.acc = self.acc.append(new, ignore_index=True)  

class robo(Automl):
    def fit(self):
        if self.auto==True:
            self.knn()
            self.logisticreg()
            self.gaussiannb()
            self.decisiontree()
            self.svm()
            self.randomforest()
            self.sgdclassifier()
        return self.acc
