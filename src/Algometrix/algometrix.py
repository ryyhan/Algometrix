import numpy as np
import pandas as pd
from .problem_types import Ptype
from .validation_regression import validation_reg
from .validation_classification import validation_class


def algometrix(
    X_train,
    X_test,
    y_train,
    y_test,
    prob_type="none",
    classification_type="None",
    algorithms="all",
    metrics="all",
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
        return validation_reg(X_train, X_test, y_train, y_test, algorithms)

    elif prob_type == "class":
        return validation_class(
            X_train, X_test, y_train, y_test, algorithms, classification_type
        )
