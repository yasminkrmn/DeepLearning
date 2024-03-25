
# -------------------------------------------------
# Prediction of Handwritten Digits - MNIST
# -------------------------------------------------

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical, plot_model

import matplotlib.pyplot as plt
import numpy as np

import warnings
from warnings import filterwarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore")

# -------------------------------------------------
# 1. Importing Data - MNIST
# -------------------------------------------------

# This is a dataset of 60,000 28x28 (784 pixels) grayscale images of the 10 digits, along with a test set of 10,000 images.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
# (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)
# (10000, 28, 28) (10000,)

np.unique(y_train)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)

# -------------------------------------------------
# 2. Plotting Images
# -------------------------------------------------
def visualize_img(data):
    plt.figure(figsize=(10,10))
    for i in range(10):
        ax = plt.subplot(5,5,i+1)
        plt.imshow(data[i], cmap='gray')
    plt.show()

visualize_img(x_train)


def pixel_visualize(image):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    width, height = image.shape

    threshold = image.max() / 2.5

    for i in range(width):
        for j in range(height):
            ax.annotate(str(round(image[i, j],2)), xy=(j,i))
            if image[i, j] > threshold:
                color = 'white'
            else:
                color = 'black'

    plt.show()

pixel_visualize(x_train[0])