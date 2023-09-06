import numpy as np
import pandas as pd
from problem_types import Ptype
from validation_regression import validation
from validation_classification import classification_models


def algometrix(
    X_train,
    X_test,
    y_train,
    y_test,
    prob_type="none",
    classification_type="none",
    algorithms="none",
    metrics="none",
    cross_validation=False,
):
    return algorithm_selection(
        X_train, X_test, y_train, y_test, prob_type, algorithms, classification_type
    )


def algorithm_selection(
    X_train, X_test, y_train, y_test, prob_type, algorithms, classification_type
):
    prob_type = Ptype(prob_type)

    if prob_type == "reg":
        return validation(X_train, X_test, y_train, y_test, algorithms)

    elif prob_type == "class":
        return classification_models(
            X_train, X_test, y_train, y_test, algorithms, classification_type
        )
