import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

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


models = [
    "AdaBoostClassifier",
    "BaggingClassifier",
    "BernoulliNB",
    "GaussianNB",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "KNeighborsClassifier",
    "LogisticRegression",
    "MultinomialNB",
    "QuadraticDiscriminantAnalysis",
    "RandomForestClassifier",
    "RidgeClassifier",
    "SVC",
]


svc = SVC(kernel="sigmoid", gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
bnb = BernoulliNB(force_alpha=True)
gnb = GaussianNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver="liblinear", penalty="l1")
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
qdc = QuadraticDiscriminantAnalysis()
rc = RidgeClassifier()
hgb = HistGradientBoostingClassifier()


clfs = {
    "SVC": svc,
    "KNeighborsClassifier": knc,
    "MultinomialNB": mnb,
    "DecisionTreeClassifier": dtc,
    "LogisticRegression": lrc,
    "AdaBoostClassifier": abc,
    "BaggingClassifier": bc,
    "ExtraTreesClassifier": etc,
    "GradientBoostingClassifier": gbdt,
    "BernoulliNB": bnb,
    "GaussianNB": gnb,
    "HistGradientBoostingClassifier": hgb,
    "QuadraticDiscriminantAnalysis": qdc,
    "RandomForestClassifier": rfc,
    "RidgeClassifier": rc,
}


def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test,y_pred)

    return accuracy


def results(X_train, X_test, y_train, y_test):
    accuracy_scores = []

    for name, clf in clfs.items():
        current_accuracy = train_classifier(clf, X_train, y_train, X_test, y_test)
        accuracy_scores.append(current_accuracy)

    performance_df = pd.DataFrame(
        {"Algorithm": clfs.keys(), "Accuracy": accuracy_scores}
    )
    print(performance_df)
