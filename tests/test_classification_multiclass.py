import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from algometrix import algometrix

df = pd.read_csv("Iris.csv")

X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)


algometrix(
    X_train,
    X_test,
    y_train,
    y_test,
    prob_type="class",
    classification_type="multiclass",
    algorithms="all",
    metrics="none",
    cross_validation=False,
)
