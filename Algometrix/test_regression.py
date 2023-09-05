import numpy as np
import pandas as pd
from algometrix import algometrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("SalaryData.csv")

X = df[["YearsExperience"]]
y = df["Salary"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
algometrix(
    X_train,
    X_test,
    y_train,
    y_test,
    prob_type="reg",
    algorithms="all",
    metrics="none",
    cross_validation=False,
)
