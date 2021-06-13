# [Convolutional Neural Networks in TensorFlow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow)

    - How to work with real-world images in different shapes and sizes.
    - Visualize the journey of an image through convolutions to understand how a computer “sees” information
    - Plot loss and accuracy, and explore strategies to prevent overfitting, including augmentation and dropout.
    - Finally, Course 2 will introduce you to transfer learning and how learned features can be extracted from models.

## Contents:
- Week 1 - [Exploring a Larger Dataset](#Exploring-a-Larger-Dataset)
- Week 2 - [Augmentation](#Augmentation)
- Week 3 - [Transfer Learning](#Transfer-Learning)
- Week 4 - [Multiclass Classifications](#Multiclass-Classifications) 

## Exploring a Larger Dataset
> [Notebook](notebooks/Course_2_Part_2_Lesson_2_Notebook.ipynb)

> https://www.kaggle.com/c/dogs-vs-cats 25K pictures of cats and dogs

```python
# Download ZIP file and extract it with python
!wget --no-check-certificate \
  https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
  -O /tmp/cats_and_dogs_filtered.zip
_____________________________________________
import os
import zipfile

local_zip = '/tmp/cats_and_dogs_filtered.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp')
zip_ref.close()
```
```py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))     
```

## Augmentation
> [Notebook](notebooks/Course_2_Part_4_Lesson_2_Notebook_(Cats_v_Dogs_Augmentation).ipynb)

> `image-augmentation` • `data-augmentation` • `ImageDataGenerator`

- All processes will happen in the main memory, from_from_directory() will generate the images on the fly. It doesn't require you to edit your raw images, nor does it amend them for you on-disk. It does it in-memory as it's performing the training, allowing you to experiment without impacting your dataset. 
- `ImageDataGenerator()` -> `flow_from_directory()` -> `fit_generator()`
- **ImageDataGenerator** will NOT add **new images** to your data set in a sense that it will not make your epochs bigger. Instead, in each epoch it will provide slightly altered images (depending on your configuration). It will always generate new images, no matter how many epochs you have.

```python
train_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=40,      # Randomly rotate image between 0 and 40°
  width_shift_range=0.2,  # Move picture inside its frame
  height_shitt_range=0.2,
  shear_range=0.2,        # Shear up to 20%
  zoom_range=0.2,         
  horizontal_flip=True,
  fill_mode='nearest')    # It attempts to recreate lost information after a transformation like a shear

train_generator = train_datagen.flow_from_directory(
        train_dir,               # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,           # Size of the batches of data, (? a number of samples per gradient update)
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,       # 2000 images = batch_size * steps, total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch
      epochs=100,
      # validation_data=validation_generator,
      # validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)
```
- https://keras.io/api/preprocessing/image/
- https://fairyonice.github.io/Learn-about-ImageDataGenerator.html
- https://keras.io/api/models/model_training_apis/#fit-method
- https://stackoverflow.com/questions/38340311/what-is-the-difference-between-steps-and-epochs-in-tensorflow
- https://stackoverflow.com/questions/51748514/does-imagedatagenerator-add-more-images-to-my-dataset

## Transfer Learning
> `inception`

> https://www.tensorflow.org/tutorials/images/transfer_learning

```python
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

# Donwload InceptionV3 weights
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = Inceptionv3(input_shape=(150, 150, 3),
                                include_top=False, # Do not include top FC (fully connected) layer
                                weights=None)
pre_trained_model.load_weights(local_weights_file) # Use own weights

# Do not retrain layers, i.e freeze them
for layer in pre_trained_model.layers:
  layer.trainable = False

# pre_trained_model.summary()

# Grab the mixed7 layer from inception, and take its output 
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Now, you'll need to add your own DNN at the bottom of these, which you can retrain to your data
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x) # Drop out 20% of neurons
x = layers.Dense(1, activation='sigmoid')(x)

# Create model using 'Model' abstract class
model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(lr=.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(...)
train_generator = train_datagen.flow_from_directory(...)
history = model.fit_generator(...)
```
> The idea behind **Dropouts** is that they **remove a random number of neurons** in your neural network. This works very well for two reasons: The first is that neighboring neurons often end up with similar weights, which can lead to overfitting, so dropping some out at random can remove this. The second is that often a neuron can over-weigh the input from a neuron in the previous layer, and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit!

## Multiclass Classifications
- Computer generated images (CGI) will help you to create a dataset. Imagine you are creating a project for detecting rock, paper, scissors (💎, 📄, ✂️) during the game. So, you need lots of images of different races for both male and female, big and little hands. 
- http://www.laurencemoroney.com/rock-paper-scissors-dataset/
<!-- 
- https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip
- https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip
- https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-validation.zip -->
- Change to `class_mode='categorical'` in flow_from_firectory(), and output Dense layer `activation='softmax'`, and loss function in model.compile `loss='categorical_crossentropy'`
- flow_from_directory() uses the alphabetical order. For example, is we test for rock the output should be [1, 0, 0] because of [rock, paper, scissors].

## Notes
- Can you use Image augmentation with Transfer Learning? 
  > Yes. It's pre-trained layers that are frozen. So you can augment your images as you train the bottom layers of the DNN with them