# [Convolutional Neural Networks in TensorFlow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow)

    - How to work with real-world images in different shapes and sizes.
    - Visualize the journey of an image through convolutions to understand how a computer “sees” information
    - Plot loss and accuracy, and explore strategies to prevent overfitting, including augmentation and dropout.
    - Finally, Course 2 will introduce you to transfer learning and how learned features can be extracted from models. 

## Exploring a Larger Dataset
> [Notebook](notebooks/deeplearning.ai-TensorFlow/Course_2_Part_2_Lesson_2_Notebook.ipynb)

> https://www.kaggle.com/c/dogs-vs-cats 25K pictures of cats and gogs

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
> [Notebook](notebooks/deeplearning.ai-TensorFlow/Course_2_Part_4_Lesson_2_Notebook_(Cats_v_Dogs_Augmentation).ipynb)

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
> https://www.tensorflow.org/tutorials/images/transfer_learning
