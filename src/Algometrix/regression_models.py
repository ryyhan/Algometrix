from .processing_regression import results
import numpy as np
import pandas as pd


def regression_models(X_train, X_test, y_train, y_test, algorithms):
    validation(X_train, X_test, y_train, y_test, algorithms)


def validation(X_train, X_test, y_train, y_test, algorithms):
    if not (isinstance(algorithms, list) or algorithms == "all"):
        raise TypeError("Parameter should be a list or 'all'")

        if not (set(algorithms).issubset(set(models))):
            raise TypeError("All elements of 'algorithms' are not in 'models'.")
        elif algorithms == "all":
            results(X_train, X_test, y_train, y_test)
    else:
        results(X_train, X_test, y_train, y_test)
