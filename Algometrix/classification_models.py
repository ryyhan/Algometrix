from processing_classification import process
import numpy as np
import pandas as pd


def classification_models(
    X_train, X_test, y_train, y_test, algorithms, classification_type
):
    validation(X_train, X_test, y_train, y_test, algorithms, classification_type)


def validation(X_train, X_test, y_train, y_test, algorithms, classification_type):
    if not (isinstance(algorithms, list) or algorithms == "all") and not (
        classification_type == "binary" or classification_type == "multi"
    ):
        raise TypeError(
            "Parameter should be a list or 'all' and type of classification must be either 'multi' or 'binary"
        )

        if not (set(algorithms).issubset(set(models))):
            raise TypeError("All elements of 'algorithms' are not in 'models'.")
        elif algorithms == "all" and (
            classification_type == "binary" or classification_type == "multi"
        ):
            process(X_train, X_test, y_train, y_test, classification_type)
    else:
        process(X_train, X_test, y_train, y_test, classification_type)
