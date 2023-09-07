# Algometrix | Your Comprehensive Machine Learning Model Comparison Tool

Algometrix is a powerful and versatile Python package designed to simplify the process of evaluating and selecting the most suitable machine-learning models for your data analysis needs. Whether you're tackling classification tasks, including binary and multiclass problems, or regression challenges, Algometrix has you covered. 

It offers extensive support for more than 22 classification algorithms/models along with six comprehensive evaluation metrics tailored towards these models. Furthermore, Algometrix caters to over 27 regression algorithms/models backed by eight diverse evaluation metrics that enable users to gain valuable insights into their models' performance.
The package facilitates seamless comparisons between multiple models while offering a broad range of standard evaluative measures such as precision, recall, F1 score, Jaccard score, mean squared error (MSE), among many others.

With its adaptable nature supporting both binary classifications as well as multiclass classifications within the realm of classifiers coupled with its proficiency in evaluating regressions tasks; Algometrix grants users unparalleled flexibility in analyzing model outputs across various domains.


## Features

- Compare over 22 classification algorithms and 27 regression models.
- Evaluate models using 14+ comprehensive metrics for classification and regression.
- Evaluation support for both binary and multiclass classification problems.
- Seamlessly process both regression and classification problems.
- Empower data-driven decisions by selecting the optimal model for your dataset.

## Installation

Install Algometrix using pip

```python
pip install Algometrix
```

For Python3, use

```python
pip3 install Algometrix
```
## Usage

**Function Definition**

```python
algometrix(X_train, X_test, y_train, y_test, prob_type="None", classification_type="None", algorithms="all", metrics="all", cross_validation=False)
```
Here,

- **X_train, X_test, y_train, y_test** are variables that contain data split into training and test.

- **prob_type** defines the type of Machine Learning Problem, i.e., *Classification* or *Regression*.

    [Takes input as "class" for classification and "reg" for regression]

- **classification_type** defines the type of Classification Problem, i.e., *Multiclass* or *Binary*.

    [Takes input as "multiclass" for Multiclass Classification and "binary" for Binary Classification]

- **algorithms** defines the models to be fitted
  
    [Takes input as "all" or a list containing the name of models (only select for the names given Details section){As of now, only "all" is available}]

- **metrics** defines the metrics to be evaluated
  
    [Takes input as "all" or a list containing the name of metrics (only select for the names given Details section){As of now, only "all" is available}]

- **cross_validation** defines if Cross Validation is to be applied
  
    [Takes input as True or False {Not Available as of now}]


### Examples

#### *Example 1 | Binary Classification*

[Example 1 | Binary Classification](https://colab.research.google.com/drive/1EYGuxj-wJGdaP--rTesa6uUPqhvp7ojH?usp=sharing)

![binary_example](https://github.com/ryyhan/Algometrix/assets/76737575/b6808fec-e91b-40a1-9788-39ef5a1bec95)

**Output:**

| | Algorithm	| Accuracy | Precision	| Recall	| F1-Score	| Jaccard	| Matthews Score |
| ---------| ---------| ---------| ---------| ---------| ---------| ---------| ---------|
|0	| SVC	| 0.416667	| 0.416667	| 0.416667	| 0.416667	| 0.263158	| 0.000000|
|1	| KNeighborsClassifier	| 0.500000	| 0.500000	| 0.500000	| 0.500000	| 0.333333	| 0.028571|
|2	| MultinomialNB	| 0.583333	| 0.583333	| 0.583333	| 0.583333	| 0.411765	| 0.377964|
|3	| DecisionTreeClassifier	| 0.333333	| 0.333333	| 0.333333	| 0.333333	| 0.200000	| -0.292770|
|4	| LogisticRegression	| 0.666667	| 0.666667	| 0.666667	| 0.666667	| 0.500000	| 0.371429|
|5	| AdaBoostClassifier	| 0.583333	| 0.583333	| 0.583333	| 0.583333	| 0.411765	| 0.169031|
|6	| BaggingClassifier	| 0.250000	| 0.250000	| 0.250000	| 0.250000	| 0.142857	| -0.478091|
|7	| ExtraTreesClassifier	| 0.250000	| 0.250000	| 0.250000	| 0.250000	| 0.142857	 | -0.507093|
|8	| GradientBoostingClassifier	| 0.333333	| 0.333333	| 0.333333	| 0.333333	| 0.200000	| -0.314286|
|9	| BernoulliNB	| 0.416667	| 0.416667	| 0.416667	| 0.416667	| 0.263158	| 0.000000|
|10	| GaussianNB	| 0.500000	| 0.500000	| 0.500000	| 0.500000	| 0.333333	| -0.028571|
|11	| HistGradientBoostingClassifier	| 0.416667	| 0.416667	| 0.416667	| 0.416667	| 0.263158	| 0.000000|
|12	| QuadraticDiscriminantAnalysis	| 0.500000	| 0.500000	| 0.500000	| 0.500000	| 0.333333	| -0.028571|
|13	| RandomForestClassifier	| 0.333333	| 0.333333	| 0.333333	| 0.333333	| 0.200000	| -0.314286|
|14	| RidgeClassifier	| 0.666667	| 0.666667	| 0.666667	| 0.666667	| 0.500000	| 0.371429|
|15	| RidgeClassifierCV	| 0.666667	| 0.666667	| 0.666667	| 0.666667	| 0.500000	| 0.371429|
|16	| Perceptron	| 0.583333	| 0.583333	| 0.583333	| 0.583333	| 0.411765	| 0.000000|
|17	| PassiveAgressiveClassifier	| 0.416667	| 0.416667	| 0.416667	| 0.416667	| 0.263158	| 0.000000|
|18	| OutputCodeClassifier	| 0.250000	| 0.250000	| 0.250000	| 0.250000	| 0.142857	| -0.507093|
|19	| MLPClassifier	| 0.583333	| 0.583333	| 0.583333	| 0.583333	| 0.411765	| 0.000000|
|20	| LogisticRegressionCV	| 0.416667	| 0.416667	| 0.416667	| 0.416667	| 0.263158	| 0.000000|
|21	| LinearDiscriminantAnalysis	| 0.666667	| 0.666667	| 0.666667	| 0.666667	| 0.500000	| 0.371429|
|22	| DummyClassifer	| 0.416667	| 0.416667	| 0.416667	| 0.416667	| 0.263158	| 0.000000|



#### *Example 2 | Multiclass Classification*

[Example 2 | Multiclass Classification - Google Colab Link](https://colab.research.google.com/drive/1ORgvb6JX6IoqbBiaxkdfAXjk5tiLr88T?usp=sharing)

![multiclass_example](https://github.com/ryyhan/Algometrix/assets/76737575/1e205040-2170-4bc9-a5a0-773425e59f0f)

**Output:**

| | Algorithm	| Accuracy | Precision	| Recall	| F1-Score	| Jaccard	| Matthews Score |
| ---------| ---------| ---------| ---------| ---------| ---------| ---------| ---------|
|0	| SVC	| 0.311111	| 0.311111	| 0.311111	| 0.311111	| 0.184211 | 0.000000|
|1 | KNeighborsClassifier	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000 |
|2 |	MultinomialNB	| 0.777778	| 0.777778	| 0.777778	| 0.777778	| 0.636364	| 0.680439|
|3	| DecisionTreeClassifier	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|4	| LogisticRegression	| 0.933333	| 0.933333	| 0.933333	| 0.933333	| 0.875000	| 0.900667|
|5	| AdaBoostClassifier	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|6	| BaggingClassifier	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|7	| ExtraTreesClassifier	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|8	| GradientBoostingClassifier	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|9	| BernoulliNB	| 0.311111	| 0.311111	| 0.311111	| 0.311111	| 0.184211	| 0.000000|
|10	| GaussianNB	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|11	| HistGradientBoostingClassifier	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|12	| QuadraticDiscriminantAnalysis	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|13	| RandomForestClassifier	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|14	| RidgeClassifier	| 0.911111	| 0.911111	| 0.911111	| 0.911111	| 0.836735	| 0.869436|
|15	| RidgeClassifierCV	| 0.911111	| 0.911111	| 0.911111	| 0.911111	| 0.836735	| 0.869436|
|16	| Perceptron	| 0.622222	| 0.622222	| 0.622222	| 0.622222	| 0.451613	| 0.548971|
|17	| PassiveAgressiveClassifier	| 0.777778	| 0.777778	| 0.777778	| 0.777778	| 0.636364	| 0.678609|
|18	| OutputCodeClassifier	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|19	| MLPClassifier	| 0.600000	| 0.600000	| 0.600000	| 0.600000	| 0.428571	| 0.431152|
|20	| LogisticRegressionCV	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|21	| LinearDiscriminantAnalysis	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000	| 1.000000|
|22	|  DummyClassifer	| 0.311111	| 0.311111	| 0.311111	| 0.311111	| 0.184211	| 0.000000|

#### *Example 3 | Regression*

[Example 3 | Regression - Google Colab Link](https://colab.research.google.com/drive/1zaeo02Gq5siwZmFVq0k6RWQ4R8QNOADs?usp=sharing)

![regression_example](https://github.com/ryyhan/Algometrix/assets/76737575/73e7b673-5c7a-4300-a674-1cc9f4c7d6af)


**Output:**

| | Algorithm	| R2 Score | MSE	| RMSE	| MAE	| MAPE	| RMSLE | MSLE | MedAE |
| ---------| ---------| ---------| ---------| ---------| ---------| ---------| ---------| ---------| ---------|
|0	| ARDRegression	| 0.917697	| 4.108494e+07	| 6409.753717	| 5239.417534	| 0.062903	| 0.077906	| 0.006069	| 4817.247851|
|1	| AdaBoostRegressor	| 0.848884	| 7.543573e+07	| 8685.374616	| 6938.123512	| 0.083586	| 0.110943	| 0.012308	| 4705.660714|
|2	| BayesianRidge	| 0.917697	| 4.108494e+07	| 6409.753717	| 5239.417534	| 0.062903	| 0.077906	| 0.006069	| 4817.247851|
|3	| CCA	| 0.921953	| 3.896024e+07	| 6241.814176	| 4967.335260	| 0.059350	| 0.076020	| 0.005779	| 4437.462764|
|4	| DecisionTreeRegressor	| 0.834468	| 8.263201e+07	| 9090.215167	| 7735.062500	| 0.097230	| 0.120206	| 0.014450	| 5588.250000|
|5	| DummyRegressor	| -0.111854	| 5.550261e+08	| 23558.992527	| 21016.659091	| 0.276376	| 0.306098	| 0.093696	| 18828.636364|
|6	| ElasticNet	| 0.904192	| 4.782645e+07	| 6915.667077	| 5911.220137	| 0.071675	| 0.084253	| 0.007099	| 6004.368917|
|7	| ExtraTreesRegressor	| 0.896412	| 5.170991e+07	| 7190.960221	| 5732.981250	| 0.073146	| 0.095814	| 0.009180	| 4867.170000|
|8	| GammaRegressor	| 0.789340	| 1.051594e+08	| 10254.726158	| 8869.015658	| 0.106939	| 0.126560	| 0.016018	| 7585.031298|
|9	| GradientBoostingRegressor	| 0.836330	| 8.170230e+07	| 9038.932528	| 7706.852903	| 0.096833	| 0.119321	| 0.014238	| 5554.335120|
|10	| HistGradientBoostingRegressor	| -0.111854	| 5.550261e+08	| 23558.992527	| 21016.659091	| 0.276376	| 0.306098	| 0.093696	| 18828.636364|
|11	| HuberRegressor	| 0.909233	| 4.531003e+07	| 6731.272606	| 5532.071087	| 0.065076	| 0.080900	| 0.006545	| 4837.661892|
|12	| IsotonicRegression	| 0.906211	| 4.681865e+07	| 6842.415671	| 5515.111954	| 0.067195	| 0.086121	| 0.007417	| 4892.522222|
|13	| KNeighborsRegressor	| 0.777560	| 1.110396e+08	| 10537.530589	| 8356.687500	| 0.101855	| 0.134679	| 0.018138	| 7351.500000|
|14	| KernelRidge	| 0.744503	| 1.275414e+08	| 11293.423112	| 9857.106177	| 0.142520	| 0.222372	| 0.049449	| 9499.519986|
|15	| Lasso	| 0.918098	| 4.088475e+07	| 6394.118608	| 5215.569586	| 0.062591	| 0.077723	| 0.006041	| 4775.106899|
|16	| LinearRegression	| 0.918098	| 4.088462e+07	| 6394.108265	| 5215.553723	| 0.062591	| 0.077723	| 0.006041	| 4775.078867|
|17	| OrthogonalMatchingPursuit	| 0.918098	| 4.088462e+07	| 6394.108265	| 5215.553723	| 0.062591	| 0.077723	| 0.006041	| 4775.078867|
|18	| PLSCanonical	| 0.921953	| 3.896024e+07	| 6241.814176	| 4967.335260	| 0.059350	| 0.076020	| 0.005779	| 4437.462764|
|19	| PLSRegression	| 0.918098	| 4.088462e+07	| 6394.108265	| 5215.553723	| 0.062591	| 0.077723	| 0.006041	| 4775.078867|
|20	| PassiveAggressiveRegressor	| -3.882800	| 2.437443e+09	| 49370.466856	| 48164.437500	| 0.601943	| 0.942270	| 0.887873	| 51737.250000|
|21	| PoissonRegressor	| 0.823058	| 8.832780e+07	| 9398.287217	| 7663.304838	| 0.093380	| 0.116128	| 0.013486	| 5834.399751|
|22	| RANSACRegressor	| 0.916163	| 4.185057e+07	| 6469.202025	| 5199.125327	| 0.061458	| 0.078296	| 0.006130	| 4447.379067|
|23	| RadiusNeighborsRegressor	| 0.824479	| 8.761805e+07	| 9360.451411	| 7125.990625	| 0.086287	| 0.116800	| 0.013642	| 4257.729167|
|24	| RandomForestRegressor	| 0.682301	| 1.585920e+08	| 12593.332585	| 9046.311605	| 0.102425	| 0.161027	| 0.025930	| 5564.875213|
|25	| Ridge	|0.916967	| 4.144903e+07	| 6438.092342	| 5281.995813	| 0.063459	| 0.078239	| 0.006121	| 4892.486577|
|26	| TheilSenRegressor	| 0.886205	| 5.680532e+07	| 7536.930405	| 6363.917428	| 0.073557	| 0.089124	| 0.007943	| 5589.457191|
|27	| TweedieRegressor	| 0.887788	| 5.601530e+07	| 7484.336856	| 6537.621805	| 0.079858	| 0.091876	| 0.008441	| 7109.469262|


## Details

### Regression Models
- ARDRegression
- AdaBoostRegressor
- BayesianRidge
- CCA
- DecisionTreeRegressor
- DummyRegressor
- ElasticNet
- ExtraTreesRegressor
- GammaRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- HuberRegressor
- IsotonicRegression
- KNeighborsRegressor
- KernelRidge
- Lasso
- LinearRegression
- OrthogonalMatchingPursuit
- PLSCanonical
- PLSRegression
- PassiveAggressiveRegressor
- PoissonRegressor
- RANSACRegressor
- RadiusNeighborsRegressor
- RandomForestRegressor
- Ridge
- TheilSenRegressor
- TweedieRegressor

### Regression Metrics
- Accuracy
- R² Score (R2Score)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Root Mean Squared Log Error (RMSLE)
- Mean Squared Log Error (MSLE)
- Median Absolute Error (MedAE)

### Classification Models
- AdaBoostClassifier
- BaggingClassifier
- BernoulliNB
- GaussianNB
- DecisionTreeClassifier
- ExtraTreesClassifier
- GradientBoostingClassifier
- HistGradientBoostingClassifier
- KNeighborsClassifier
- LogisticRegression
- MultinomialNB
- QuadraticDiscriminantAnalysis
- RandomForestClassifier
- RidgeClassifier
- RidgeClassifierCV
- SVC
- Perceptron
- PassiveAgressiveClassifier
- OutputCodeClassifier
- MLPClassifier
- LogisticRegressionCV
- LinearDiscriminantAnalysis
- DummyClassifier

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Jaccard Score
- Matthews Score

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributor

[Rehan Khan](https://github.com/ryyhan)

## Acknowledgement

Special thanks to the open-source community for their contributions to machine learning and data science.
