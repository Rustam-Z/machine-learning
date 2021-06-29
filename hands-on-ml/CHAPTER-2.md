## CHAPTER 2: End-to-End Machine Learning Project
1. Frame the problem and look at the big picture.
2. Get the data.
3. Discover and visualize the data to gain insights.
4. Prepare the data for Machine Learning algorithms.
5. Explore many different models and shortlist the best ones.
6. Fine-tune your models and combine them into a great solution.
7. Present your solution.
8. Launch, monitor, and maintain your system.

### Look at the Big Picture
- **Frame the Problem**
    - **Question 1:** What exactly is the business objective, maybe no need to ML? How does the company expect to use and benefit from this model?
    - You must understand the problem, because it will help to you select the algorithm, and performance measure. You can build the pipeline.
    - **Question 2:** Is there any solutions?
    - Now design ML system. Is it supervised, unsupervised, what kind of task it is (classification, regression)? Whether apply batch or online learning?
- **Select a Performance Measure** - loss function, for regression task Root Mean Squared Error is used, it gives the idea of how much error the system typically makes in its predictions, with a higher weight for large errors.
    - **RMSE** - Root Mean Squared Error, Euclidean norm or L2 norm. The higher the norm index, the more it focuses on large values and neglects small ones. But when outliers are exponentially rare RMSE is better.
    - **MSE** - Mean Squared Error, Manhattan norm or L1 norm, **not so sensitive to outliers as RMSE**
- **Check the Assumptions** - understand input & output of model, problem & solution

### Get the data
- **Take a Quick Look at the Data Structure**
    - Analyse outputs of `df.info(), df.describe(), df.hist()`
- **Create a Test Set** - put it aside, and never look at it.
    - `split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`

### Discover and Visualize the Data to Gain Insights
- Look at the correlations between features

### Prepare the Data for Machine Learning Algorithms
- Data cleaning, missing values
    1. Get rid of the corresponding districts.
    2. Get rid of the whole attribute.
    3. Set the values to some value (zero, themean, themedian, etc.).
    ```py
    housing.dropna(subset=["total_bedrooms"])    # option 1
    housing.drop("total_bedrooms", axis=1)       # option 2
    median = housing["total_bedrooms"].median()  # option 3
    housing["total_bedrooms"].fillna(median, inplace=True)
    ```
    ```    
    Axis 0 will act on all the ROWS in each COLUMN    
    Axis 1 will act on all the COLUMNS in each ROW
    ```    
    ```py
    df1 = df.select_dtypes(include=[np.number]) # to select numerical attributes
    df2 = df.select_dtypes(include=['object']) # for categorical attributes

    ```
