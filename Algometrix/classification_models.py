import numpy as np
import pandas as pd

models = ['AdaBoostClassifier',
    'BaggingClassifier',
    'BernoulliNB',
    'CalibratedClassifierCV',
    'GaussianNB',
    'ExtraTreeClassifier',
    'DecisionTreeClassifier',
    'ExtraTreesClassifier',
    'GaussianProcessClassifier',
    'GradientBoostingClassifier',
    'HistGradientBoostingClassifier',
    'KNeighborsClassifier',
    'LabelPropagation',
    'LabelSpreading',
    'LinearDiscriminantAnalysis',
    'LinearSVC',
    'LogisticRegression',
    'LogisticRegressionCV',
    'MLPClassifier',
    'MultinomialNB',
    'NearestCentroid',
    'NuSVC',
    'PassiveAggressiveClassifier',
    'QuadraticDiscriminantAnalysis',
    'RandomForestClassifier',
    'RidgeClassifier',
    'RidgeClassifierCV',
    'SGDClassifier',
    'SVC',]

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