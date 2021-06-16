# Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow

Rustam-Z🚀 • 29 May 2021

Hello, I am Rustam. Here, you can find my notes on `Hands-on Machine Learning` book. Use `TIP!` in searching bar to find tips while developing ML system. 

## Roadmap
- Part I, *The Fundamentals of Machine Learning*, end-to-end ML process
    - Handling, cleaning, and preparing data.
    - Selecting and engineering features.
    - Learning by fitting a model to data.
    - Optimizing a cost function.
    - Selecting a model and tuning hyperparameters using cross-validation.
    - Underfitting and overfitting (the bias/variance tradeoff).
    - Unsupervised learning techniques: clustering, density estimation and anomaly detection.
    - Algorithms: Linear and Polynomial Regression, Logistic Regression, k-Nearest Neighbors, Support Vector Machines, Decision Trees, Random Forests, and Ensemble methods.

## Contents
- [CHAPTER 1: The Machine Learning Landscape](#CHAPTER-1:-The-Machine-Learning-Landscape)
    - What is Machine Learning?
    - Why use ML?
    - Types of ML Systems
    - Main Challenges of ML
    - Testing and Validating
- [CHAPTER 2: End-to-End Machine Learning Project](CHAPTER-2:-End-to-End-Machine-Learning-Project)
    - Look at the Big Picture
    - Get the data

## CHAPTER 1: The Machine Learning Landscape
### What is Machine Learning?
- Machine learning (ML) is field of study that gives computers the ability to learn without being explicitly programmed.
- A computer program is said to learn from *experience E* with respect to some *task T* and some *performance measure P*, if its performance on T, as measured by P, improves with experience E.
- **Example:** T = flag spam for new emails, E = the training data, P = accuracy, the ratio of correctly classified emails.

### Why use ML?
- Problems for which existing solutions require a lot of hand-tuning or long lists of
rules: one Machine Learning algorithm can often simplify code and perform bet‐
ter. (spam classifier)
- Complex problems for which there is no good solution at all using a traditional
approach: the best Machine Learning techniques can find a solution. (speech recognition)
- Fluctuating environments: a Machine Learning system can adapt to new data.
- Getting insights about complex problems and large amounts of data. (data mining)

### Types of ML Systems
- Whether or not they are trained with human supervision `supervised, unsupervised, semisupervised, and Reinforcement Learning`
- Whether or not they can learn incrementally on the fly `online vs batch learning`
- Whether they work by simply comparing new data points to known data points, or instead detect patterns in the training data and build a predictive model, much like scientists do `instance-based vs model-based learning`

- **Supervised learning** - training data with labels (expected outputs). 
    - Tasks: classification, regression (univariate / multivariate). 
    - Class / sample / label / feature (predictors: age, brand, ...) / attribute
    - **Algorithms**
        - k-Nearest Neighbors
        - Linear Regression
        - Logistic Regression
        - Support Vector Machines (SVMs)
        - Decision Trees and Random Forests
        - Neural networks2

- **Unsupervised learning** - training data is unlabeled.
    - Tasks: clustering, anomaly detection, visualization & dimensionality reduction. 
    - Clustering (find similar visitors)
        - K-Means
        - DBSCAN
        - Hierarchical Cluster Analysis (HCA)
    - Anomaly detection & novelty detection (detect unusual things)
        - One-class SVM
        - Isolation Forest
    - Visualization and dimensionality reduction (king of feature extraction)
        - Principal Component Analysis (PCA)
        - Kernel PCA
        - Locally-Linear Embedding (LLE)
        - t-distributed Stochastic Neighbor Embedding (t-SNE)
    - Association rule learning
        - Apriori
        - Eclat

    - `TIP!` Use dimensionality reduction algo before feeding to supervised learning algorithm.
    - `TIP!` Automatically removing outliers from a dataset before feeding it to another learning algorithm.

- **Semisupervised learning** - a lot of unlabeled data and a little bit of labeled data. 
    - Example: like in Google photos, it recongnizes same person in many pictures. We need supervised part because we need to seperate similar clusters. (like similar people)

- **Reinforcement Learning** - *agent* can observe environment, and perform some actions, and get *rewards* and *penalties*. Then it must teach itself the best strategy (*policy*) to get max reward. A policy defines what action the agent should choose when it is in a given situation.
<br><img src="img/reinforcement-learning.png" width=450px center>

- **Batch learning** - or *offline learning*, when you have new type of data, you need to retrain over whole dataset every time.
- **Online learning** - you train the system incrementally on a new data or mini-batch of data. 
    - You must set *learning rate* parameter, if you set hugh rate, then your system rapidly adapt to new data, but it will tend to forget the old data. 
    - A big challenge if bad data is fed to the system, the system’s performance will gradually decline. 
    - `TIP!` Monitor your latest input data using an anomaly detection algorithm.

- **Instance-based learning** - the system learns the examples by heart, then generalizes to new cases by comparing them to the learned examples using a *similarity measure*.
- **Model-based learning** - build the model, then use it to make *predictions*.

### Main Challenges of ML
- “Bad algorithm” and “bad data”
- **Bad data**
- If some instances are missing a few features (e.g., 5% of your customers did not specify their age), you must decide whether you want to ignore this attribute altogether, ignore these instances, fill in the missing values (e.g., with the median age), or train one model with the feature and one model without it, and so on.
- **Feature engineering**, involves:
    - *Feature selection*: selecting the most useful features to train on among existing features.
    - *Feature extraction*: combining existing features to produce a more useful one (dimensionality reduction algorithms can help).
    - Creating new features by gathering new data.

- **Bad algorithm**
    - **Overfitting** means that the model performs well on the training data, but it does not generalize well. How to overcome?
        - To simplify the model by selecting one with fewer parameters (a linear model rather than a high-degree polynomial model), by redusing number features in training data or or by constraining the model (with regularization).
        - To gather more training data.
        - To reduce the noise in the training data (fix data errors and remove outliers).
    - **Underfitting** occurs when your model is too simple to learn the underlying structure of the data. The options to fix:
        - Selecting a more powerful model, with more parameters.
        - Feeding better features to the learning algorithm (feature engineering)
        - Reducing the constraints on the model (reducing the regularization hyperparameter)

- The system will not perform well if your training set is too small, or if the data is not representative (production level data), noisy, or polluted with irrelevant features (garbage in, garbage out). Lastly, your model needs to be neither too simple.

### Testing and Validating
- 80% training and 20% testing. If 10 million samples 1% for testing is enough.
- **Hyperparameter Tuning and Model Selection** `page 32`
    - Example: you are hesiteting between two models linear and polinomial. You must try both and see which one is generalizing better on test set. You want to apply regularization to decrease overfitting, so you don't know how to choose a hyperparameter. Try 100 different hyperparameters, and find the best which produces small error.
    - However, after you deployed your model you see 15% error. It is probably because you chose *hp* for this particular set. Then you should use **holdout validation "with validation / dev set"**. You train multiple models with various hyperparameters on the reduced training set (training - validation set). Select model performing best on val-on set. And train again on full dataset. 
    - [**Cross validation**](https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/)
- **Data Mismatch** `page 33`
    - Example: You want to developer flowers species classifier. You downloaded pictures from web. And you have 10K pictures taken with the app. **TIP! Remember, your validation and test set must be as representitive as possible you expect to use in production.** In this case divide 50 / 50 to dev & test sets (pics must not be duplicated in both, even near-duplicate). 
    - After training you see that model on validation set is very poor. Is it overfitting or mismatch between web and phone pics?
    - One solution, is to take the part of training (web pics) into **train-dev set**. After training a model, you see that model on train-dev set is good. Then the problem is data mismatch. Use preprocessing, and make web pics look like phone pics. 
    - But if model is bad on train-dev set, then you have overfitting. You should try to simplify or regularize the model, get more training data and clean up the training data.
    
### Extra 
- **Hyper-parameters** are those which we supply to the model, for example: number of hidden Nodes and Layers, input features, Learning Rate, Activation Function etc in Neural Network, while **Parameters** are those which would be learnt during training by the machine like Weights and Biases.


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
- **Download the Data**
- **Take a Quick Look at the Data Structure**
    - Analyse outputs of `df.info(), df.describe(), df.hist()`
- **Create a Test Set** - put it aside, and never look at it.