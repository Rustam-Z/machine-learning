## Machine Learning

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
-->

<!-- 
So, here I will include:
End-to-End Machine Learning Project step by step code snippets and notes, section by section. When I will create new project, so that I can refer to any section I stuck.
-->
### End-to-End Machine Learning Project
- Look at the Big Picture
    - Frame the problem
    - Performance measure
- Get the data, create test set
- Discover data, look at the correlations
- Prepare data for ML algorithms
    - Data cleaning (numerical, categorical)
    - Feature scaling
    - Pipelines
- Select and train a model
    - Evaluation on training set
    - Cross-validation
- Fine-tune model
    - Ensemble models
    - Evaluate on test set
- Launch and monitor

```py
"""Transforming continuous numerical attributes to categorical"""
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```
