import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.linear_model import Ridge


class Magic:
    """ This class want to apply different ML-methods on a certain data """
    
    def __init__(self):
        self.data = None
        self.clean_data = None
                
    def load_csv(self, path):
        self.data = pd.read_csv(path)

        return self.data
    
    def cleanup(self, data_name):
        self.clean_data = data_name.dropna(axis=0, how='any')
        
        return self.clean_data
    
    def categorize(self, column_name):
        col_rename = []
        data_cat_encoded, data_cat_categories = self.data[column_name].factorize()
        encoder = OneHotEncoder()
        data_cat_1hot = encoder.fit_transform(data_cat_encoded.reshape(-1,1))
        data_cat_1hot_array = data_cat_1hot.toarray()      
        new_cat = pd.DataFrame(data = data_cat_1hot_array, columns = list(data_cat_categories))    # Mein neues DF mit der Kategorisieriung
        cat_join = self.data.join(new_cat, how='outer') # Ursprüngliches DataFrame wird mit dem DF verknüpft
    
        return cat_join.drop(column_name, axis = 1)
        
    def scale_data(self, X_train, X_test):  
        """ Scale your input data. It is a function in _Train_Test"""
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        return scaler.transform(X_train), scaler.transform(X_test)

    def Train_Test(self, X, y):
        """ Train_Test_Split"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, 
                                                                                    random_state = 42,
                                                                                   test_size = 0.20)
        return self.X_train, self.X_test, self.y_train, self.y_test
        
   # def cross_val(self):

        
    def apply_reg_model(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {
                "linear_regression": { 'function' : LinearRegression(), 'metrics' : ["rmse"]},
                "kneighbor_regression": { 'function' : KNeighborsRegressor(), 'metrics' : ["rmse"]},
                "decision_tree_regressor": { 'function' : DecisionTreeRegressor(), 'metrics' : ["rmse"]},
                "random_forest_regressor": { 'function' : RandomForestRegressor(), 'metrics' : ["rmse"]}
            }
        
        return self._apply_model()
        
    def apply_class_model(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {
                "logistic_regression": { 'function' : LogisticRegression(d), 'metrics' : ["logloss"]},
                "kn_classifier": { 'function' : KNeighborsClassifier(), 'metrics' : []}, 
                "SVC": { 'function' : SVC(), 'metrics': []},
                "neural network": { 'function' : MLPClassifier(), 'metrics': ["logloss"]},
                "decision tree": { 'function' : DecisionTreeClassifier(), 'metrics' : []},
                "random forest": { 'function' : RandomForestClassifier(), 'metrics' : []},
                "gradient boosting": { 'function' : GradientBoostingClassifier(), 'metrics' : []}                 
            }
        return self._apply_model()
    
    #def apply_clustering_model(self):
        
        
    def _apply_model(self):
        result = {}
        for key, model in self.models.items():
            model_f = model['function']
            result[key] = {}
            model_f.fit(self.X_train, self.y_train)
            y_pred = model_f.predict(self.X_test)
            result[key]["y_pred"] = y_pred
            result[key]["score"] = model_f.score(self.X_test, self.y_test)
            if "rmse" in model["metrics"]:
                result[key]["rmse"] = np.sqrt(mean_squared_error(self.y_test, y_pred))
            if "logloss" in model["metrics"] :
                result[key]["logloss"] = log_loss(self.y_test, model_f.predict_proba(self.X_test))
            
        return result
        
    def confusion_matrix(self, y_test, *, y_pred, score, **kwargs):
        fig = plt.figure(figsize=(9,9))
        cm = metrics.confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, square = True, fmt=".2f")
        title = 'Accuracy Score: {0}'.format(score)
        plt.title(title, size = 18)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
