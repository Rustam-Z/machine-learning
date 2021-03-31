# [Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](https://www.coursera.org/learn/introduction-tensorflow/home/welcome)

## Contents:
- Week 1 - A new programming paradigm
- Week 2 - Intro to CV
- Week 3 - CNN
- Week 4 - Using real-world images
 
> `!pip install tensorflow==2.0.0-alpha0` run it to use TensorFlow 2.x in Google Colab

## A primer in machine learning
<img src="img/1.png" width=400/>

## The ‘Hello World’ of neural networks
```python
from keras import models
from keras import layers
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error') # Guess the pattern and measure how badly or good the algorithm works

# Just imagine you have lots of Xs and Ys, the computer doesn't know the correlation between them. Your algorithm tries to connect Xs to Ys (makes guesses). The loss functions looks at the predicted outputs and actial outputs and *measures how good or badly the guess was. Then it gives its value to optimizer which figures out the next guess (update its parameters). So the optimizer thinks about how good or how badly the guess was done using the data from the loss function.

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500) # Training

print(model.predict([10.0])) # You can expect 19 because y = 2x - 1, but it will be very close to ≈19
```