import numpy as np
import pandas as pd

from sklearn import metrics

from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_decomposition import CCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import TheilSenRegressor


models = [
    "ARDRegression",
    "AdaBoostRegressor",
    "BayesianRidge",
    "CCA",
    "DecisionTreeRegressor",
    "DummyRegressor",
    "ElasticNet",
    "ExtraTreesRegressor",
    "GammaRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "HuberRegressor",
    "IsotonicRegression",
    "KNeighborsRegressor",
    "KernelRidge",
    "Lasso",
    "LinearRegression",
    "OrthogonalMatchingPursuit",
    "PLSCanonical",
    "PLSRegression",
    "PassiveAggressiveRegressor",
    "PoissonRegressor",
    "RANSACRegressor",
    "RadiusNeighborsRegressor",
    "RandomForestRegressor",
    "Ridge",
    "TheilSenRegressor",
    "TweedieRegressor",
]


# regression
ardr = linear_model.ARDRegression()
adbr = AdaBoostRegressor(random_state=0, n_estimators=50)
brr = linear_model.BayesianRidge()
cca = CCA(n_components=1)
dtr = DecisionTreeRegressor(random_state=0)
dummy = DummyRegressor(strategy="mean")
enet = ElasticNet(random_state=0)
xtr = ExtraTreesRegressor(n_estimators=100, random_state=0)
gmr = linear_model.GammaRegressor()
gbr = GradientBoostingRegressor(random_state=0)
hist = HistGradientBoostingRegressor()
huber = HuberRegressor()
iso = IsotonicRegression()
knn = KNeighborsRegressor(n_neighbors=2)
krr = KernelRidge(alpha=1.0)
lasso = linear_model.Lasso(alpha=0.1)
lr = LinearRegression()
omp = OrthogonalMatchingPursuit()
plsca = PLSCanonical(n_components=1)
plsr = PLSRegression(n_components=1)
par = PassiveAggressiveRegressor(max_iter=50, random_state=0)
pr = linear_model.PoissonRegressor()
ransac = RANSACRegressor(random_state=0)
rnr = RadiusNeighborsRegressor(radius=1.0)
rfr = RandomForestRegressor(max_depth=2, random_state=0)
ridge = Ridge(alpha=1.0)
tsr = TheilSenRegressor(random_state=0)
twee = linear_model.TweedieRegressor()


regs = {
    "ARDRegression": ardr,
    "AdaBoostRegressor": adbr,
    "BayesianRidge": brr,
    "CCA": cca,
    "DecisionTreeRegressor": dtr,
    "DummyRegressor": dummy,
    "ElasticNet": enet,
    "ExtraTreesRegressor": xtr,
    "GammaRegressor": gmr,
    "GradientBoostingRegressor": gbr,
    "HistGradientBoostingRegressor": hist,
    "HuberRegressor": huber,
    "IsotonicRegression": iso,
    "KNeighborsRegressor": knn,
    "KernelRidge": krr,
    "Lasso": lasso,
    "LinearRegression": lr,
    "OrthogonalMatchingPursuit": omp,
    "PLSCanonical": plsca,
    "PLSRegression": plsr,
    "PassiveAggressiveRegressor": par,
    "PoissonRegressor": pr,
    "RANSACRegressor": ransac,
    "RadiusNeighborsRegressor": rnr,
    "RandomForestRegressor": rfr,
    "Ridge": ridge,
    "TheilSenRegressor": tsr,
    "TweedieRegressor": twee,
}


def train_regressor(reg, X_train, y_train, X_test, y_test):
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    accuracy = metrics.r2_score(y_test, y_pred)
    # precision = precision_score(y_test,y_pred)

    return accuracy


def results(X_train, X_test, y_train, y_test):
    accuracy_scores = []

    for name, reg in regs.items():
        current_accuracy = train_regressor(reg, X_train, y_train, X_test, y_test)
        accuracy_scores.append(current_accuracy)

    performance_df = pd.DataFrame(
        {"Algorithm": regs.keys(), "Accuracy": accuracy_scores}
    )
    print(performance_df)
