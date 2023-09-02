import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

models = ['AdaBoostClassifier',
    'BaggingClassifier',
    'BernoulliNB',
    'GaussianNB',
    'DecisionTreeClassifier',
    'ExtraTreesClassifier',
    'GradientBoostingClassifier',
    'HistGradientBoostingClassifier',
    'KNeighborsClassifier',
    'LogisticRegression',
    'MultinomialNB',
    'QuadraticDiscriminantAnalysis',
    'RandomForestClassifier',
    'RidgeClassifier',
    'SVC']

    
def classification_models(algorithms):
    validation(algorithms)

    

def validation(algorithms):
    if not (isinstance(algorithms, list) or algorithms == "all"):
        raise TypeError("Parameter should be a list or 'all'")

        if not(set(algorithms).issubset(set(models))):
            raise TypeError("All elements of 'algorithms' are not in 'models'.")
        elif algorithms == "all":
            return "OK" 
    else:
        return "OK"

