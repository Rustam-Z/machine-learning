import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('/Users/rustamz/machine-learning/machine-learning-area/tensorflow-in-practice/notebooks/Exercises/my_model.h5')

image = cv2.imread('/Users/rustamz/machine-learning/machine-learning-area/tensorflow-in-practice/img/3.jpg')
image = cv2.resize(image,(28,28))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
data = np.vstack([gray])
data=data/255.0

plt.imshow(gray, cmap='gray')
plt.show()

indices_one = data == 1
data[indices_one] = 0 # replacing 1s with 0s
print(data)

predictions = model.predict(np.expand_dims(data, 0))
print("\nAnswer:")
print(predictions)
