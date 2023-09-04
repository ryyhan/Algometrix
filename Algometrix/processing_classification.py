import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import matthews_corrcoef

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



def train_classifier_multiclass(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1 = f1_score(y_test, y_pred, average="micro")
    jaccard = jaccard_score(y_test, y_pred, average="micro")
    matthews = matthews_corrcoef(y_test, y_pred)

    return accuracy, precision, recall, f1, jaccard, matthews


def results_multiclass(X_train, X_test, y_train, y_test):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    jaccard_scores = []
    matthews_scores = []

    for name, clf in clfs.items():
        (
            current_accuracy,
            current_precision,
            current_recall,
            current_f1,
            current_jaccard,
            current_matthews,
        ) = train_classifier_multiclass(clf, X_train, y_train, X_test, y_test)
        accuracy_scores.append(current_accuracy)
        precision_scores.append(current_precision)
        recall_scores.append(current_recall)
        f1_scores.append(current_f1)
        jaccard_scores.append(current_jaccard)
        matthews_scores.append(current_matthews)

    performance_df = pd.DataFrame(
        {
            "Algorithm": clfs.keys(),
            "Accuracy": accuracy_scores,
            "Precision": precision_scores,
            "Recall": recall_scores,
            "F1-Score": f1_scores,
            "Jaccard": jaccard_scores,
            "Matthews Score": matthews_scores,
        }
    )
    print(performance_df)


def train_classifier_binary(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1 = f1_score(y_test, y_pred, average="micro")
    jaccard = jaccard_score(y_test, y_pred, average="micro")
    matthews = matthews_corrcoef(y_test, y_pred)

    return accuracy, precision, recall, f1, jaccard, matthews


def results_binary(X_train, X_test, y_train, y_test):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    jaccard_scores = []
    matthews_scores = []

    for name, clf in clfs.items():
        (
            current_accuracy,
            current_precision,
            current_recall,
            current_f1,
            current_jaccard,
            current_matthews,
        ) = train_classifier_binary(clf, X_train, y_train, X_test, y_test)
        accuracy_scores.append(current_accuracy)
        precision_scores.append(current_precision)
        recall_scores.append(current_recall)
        f1_scores.append(current_f1)
        jaccard_scores.append(current_jaccard)
        matthews_scores.append(current_matthews)

    performance_df = pd.DataFrame(
        {
            "Algorithm": clfs.keys(),
            "Accuracy": accuracy_scores,
            "Precision": precision_scores,
            "Recall": recall_scores,
            "F1-Score": f1_scores,
            "Jaccard": jaccard_scores,
            "Matthews Score": matthews_scores,
        }
    )
    print(performance_df)




def process(X_train, X_test, y_train, y_test, classification_type):
    if classification_type == "multiclass":
        results_multiclass(X_train, X_test, y_train, y_test)

    if classification_type == "binary":
        results_binary(X_train, X_test, y_train, y_test)