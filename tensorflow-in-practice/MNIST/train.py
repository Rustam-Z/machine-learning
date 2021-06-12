import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.90):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

image = cv2.imread('3.png')
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

model.save('my_model.h5')