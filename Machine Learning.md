# Machine Learning

    - Handling, cleaning, and preparing data.
    - Selecting and engineering features.
    - Learning by fitting a model to data.
    - Optimizing a cost function.
    - Selecting a model and tuning hyperparameters using cross-validation.
    - Underfitting and overfitting (the bias/variance tradeoff).
    - Unsupervised learning techniques: clustering, density estimation and anomaly detection.
    - Algorithms: Linear and Polynomial Regression, Logistic Regression, k-Nearest Neighbors, Support Vector Machines, Decision Trees, Random Forests, and Ensemble methods.
<!-- 
it should include:
- important tips, notes
- code snippets which are usefull when starting new projects
- pipelines, ml models codes

So, here I will include:
End-to-End Machine Learning Project step by step code snippets and notes, section by section. When I will create new project, so that I can refer to any section I stuck.
-->

## End-to-End Machine Learning Project
- Frame the problem and look at the big picture
    - Goal and Performance measure
- Get the data
    - [Create test set](#Create-test-set)
- Explore the data to gain insights (EDA)
    - [Looking for correlations](#Looking-for-Correlations)
    - Experimenting with attribute combinations
- Prepare data for ML algorithms
    - [Data cleaning](#Data-cleaning)
    - [Handling text and categorical attributes](#Handling-text-and-categorical-attributes)
    - Feature Scaling
    - Transformation Pipelines
- Fine-tune your models and combine them into a great solution
    - Training and evaluation on training set
    - Cross-validation
- Fine-Tune model
    - Grid Search
    - Randomized Search
    - Ensemble models
    - Evaluate on test set
- Launch and monitor

## Get the data
### Create test set
```py
'''Create test set'''
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
```
```py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```
```py
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 0, 1, 1, 1])
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
sss.get_n_splits(X, y) 

for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

## Explore the data to gain insights
```py
'''Visualizing Geographical Data'''
data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
```
```py
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
```
### Looking for Correlations
```py
'''Looking for Correlations'''
corr_matrix = data.corr()
corr_matrix["any_column"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```
```py
# Correlations between features
all_data_corr = all_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
all_data_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
all_data_corr.drop(all_data_corr.iloc[1::2].index, inplace=True)
all_data_corr_nd = all_data_corr.drop(all_data_corr[all_data_corr['Correlation Coefficient'] == 1.0].index)

corr = all_data_corr_nd['Correlation Coefficient'] > 0.1
all_data_corr_nd[corr]
```
```py
# pivot_table() vs groupby(), the below lines are the same
pd.pivot_table(df, index=["a"], columns=["b"], values=["c"], aggfunc=np.sum)
df.groupby(['a','b'])['c'].sum()
```
```py
# Aggregate using one or more operations over the specified axis
# agg()-can be applied to multiple groups together
df.agg(['sum', 'min'])
df_all.groupby(['Sex', 'Pclass']).agg(lambda x:x.value_counts().index[0])['Embarked'] 

# Apply a function along an axis of the DataFrame
# apply()-cannot be applied to multiple groups together 
df.apply(np.sqrt)
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
```

## Prepare data for ML algorithms
- https://stackoverflow.com/questions/48673402/how-can-i-standardize-only-numeric-variables-in-an-sklearn-pipeline
### Data Cleaning
```py
housing.dropna(subset=["total_bedrooms"])    # Get rid of the corresponding districts
housing.drop("total_bedrooms", axis=1)       # Get rid of the whole attribute
median = housing["total_bedrooms"].median()  # Set the values to some value (zero, mean, median)
housing["total_bedrooms"].fillna(median, inplace=True)
```
```py
'''SimpleImputer, filling with the missing numerical attributes with the "median"'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number]) # just numerical attributes
imputer.fit(housing_num) # "trained" inputer, now it is ready to transform the training set by replacing missing values with the learned medians
imputer.statistics_ # same as "housing_num.median().values"
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index) # new dataframe
```

### Handling text and categorical attributes
- [select_dtypes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html)
```py
'''Transforming continuous numerical attributes to categorical'''
housing["income_cat"] = pd.cut(housing["median_income"], 
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                                labels=[1, 2, 3, 4, 5])
```
