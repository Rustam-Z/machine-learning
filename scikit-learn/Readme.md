# Scikit-Learn

- [freeCodeCamp.org](https://youtu.be/pqNCD_5r0IU)

### How to save / upload model
```py
import joblib

model = joblib.load('model.sav') # Load the model
joblib.dump(model, 'model.sav') # Save the model
```

### KNN
> [Notebook](knn.ipynb)
- Measured with Euclidean or Manhattan [distance](https://www.analyticsvidhya.com/blog/2020/02/4-types-of-distance-metrics-in-machine-learning/)
- For **KNN regressor** you take the average of `n_neighbors=23` nearest neighbours
- For **KNN classifier** you take the mood of `n_neighbors=23` nearest neighbours

### SVM
> [Notebook](svm.ipynb)
- `support vectors`, `hyperplane`, `margin`, `linear seperable`, `non-linear seperable`
- Our goal is to **maximize** the **margin** (distance between marginal hyperplanes)
- **SVM kernels** - transforms from low-dimension to high-dimension