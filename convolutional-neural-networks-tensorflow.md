# [Convolutional Neural Networks in TensorFlow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow)

- How to work with real-world images in different shapes and sizes.
- Visualize the journey of an image through convolutions to understand how a computer “sees” information
- Plot loss and accuracy, and explore strategies to prevent overfitting, including augmentation and dropout.
- Finally, Course 2 will introduce you to transfer learning and how learned features can be extracted from models. 

## Exploring a Larger Dataset
> [Notebook](notebooks/deeplearning.ai-TensorFlow/Course_2_Part_2_Lesson_2_Notebook.ipynb)

> https://www.kaggle.com/c/dogs-vs-cats

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
