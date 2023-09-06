import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from processing_regression import results

df = pd.read_csv("SalaryData.csv")

X = df[["YearsExperience"]]
y = df["Salary"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

answer = results(
    X_train,
    X_test,
    y_train,
    y_test,
)

print(answer)
