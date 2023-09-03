import numpy as np
import pandas as pd
from problem_types import Ptype
from regression_models import regression_models
from classification_models import classification_models


def algometrix(
    X_train,
    X_test,
    y_train,
    y_test,
    prob_type="none",
    algorithms="none",
    metrics="none",
    cross_validation=False,
):
    algorithm_selection(X_train, X_test, y_train, y_test, prob_type, algorithms)


def algorithm_selection(X_train, X_test, y_train, y_test, prob_type, algorithms):
    prob_type = Ptype(prob_type)
    print(prob_type)
    print(algorithms)

    if prob_type == "reg":
        regression_models(X_train, X_test, y_train, y_test, algorithms)
    elif prob_type == "class":
        classification_models(X_train, X_test, y_train, y_test, algorithms)
