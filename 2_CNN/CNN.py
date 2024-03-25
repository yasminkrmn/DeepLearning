# -------------------------------------------------
# Prediction of Garbage
# -------------------------------------------------

import cv2
import urllib
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random, os, glob
from imutils import paths
from sklearn.utils import shuffle
from urllib.request import urlopen

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix, classification_report


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# -------------------------------------------------
# 1. Importing Data
# -------------------------------------------------

dir_path = '2_CNN/dataset-resized'
target_size = (224, 224)
waste_labels = {'cardboard':0, 'glass':1, 'metal':2, 'paper':3, 'plastic':4, 'trash':5}
def load_images(dir_path, target_size):
    """
    Load images and their corresponding labels from the specified directory.

    Parameters:
    - dir_path (str): Path to the directory containing the image dataset.
    - target_size (tuple): Target size to which the images will be resized.

    Returns:
    - x (list): List of image arrays.
    - labels (list): List of corresponding labels for the images.
    """
    x = []  # List to store image arrays
    labels = []  # List to store corresponding labels

    # Loop through the images in the directory
    for img_path in sorted(list(paths.list_images(dir_path))):
        # Read the image using OpenCV
        img = cv2.imread(img_path)
        # Resize the image to the target size
        img = cv2.resize(img, target_size)
        # Append the resized image to the list
        x.append(img)

        # Extract the label from the image path
        label = img_path.split(os.path.sep)[-2]
        # Map the label to its corresponding numeric value
        labels.append(waste_labels[label])

    # Shuffle the data
    x, labels = shuffle(x, labels, random_state=42)

    # Print information about the loaded data
    print(f'X shape: {np.array(x).shape}')
    print(f'Label count: {len(np.unique(labels))} Obs number: {len(labels)}')

    return x, labels

x, labels= load_images(dir_path, target_size)
# X shape: (2527, 224, 224, 3)
# Label count: 6 Obs number: 2527

input_shape = (np.array(x[0]).shape[1], np.array(x[0]).shape[1], 3)
print(input_shape)
# (224, 224, 3)

# -------------------------------------------------
# 2. Visualizing Images
# -------------------------------------------------

def visualize_img(image_batch, label_batch):
    """
    Visualize a batch of images along with their corresponding labels.

    Parameters:
    - image_batch (numpy.ndarray): Batch of image arrays.
    - label_batch (numpy.ndarray): Batch of corresponding label arrays.

    Returns:
    None
    """
    plt.figure(figsize=(10, 10))
    for i in range(10):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(image_batch[i])
        plt.title(np.array(list(waste_labels.keys()))[
                      to_categorical(label_batch, num_classes=6)[i] == 1][0].title())
        plt.axis('off')
    plt.show()


visualize_img(x, labels)

# -------------------------------------------------
# 3. Data Preparation
# -------------------------------------------------

# This code configures an ImageDataGenerator in TensorFlow Keras to perform data augmentation
# and scaling operations, including horizontal and vertical flips, validation data splitting,
# rescaling pixel values, shear transformation, zooming, and shifting in both width and height.

train = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test = ImageDataGenerator(rescale=1./255,
                          validation_split=0.1)

train_generator = train.flow_from_directory(
    directory=dir_path, target_size=(target_size),
    class_mode='categorical',
    subset='training'
)
test_generator = test.flow_from_directory(
    directory=dir_path, target_size=(target_size),
    batch_size=251,
    class_mode='categorical',
    subset='validation'
)


# -------------------------------------------------
# 4. Modeling
# -------------------------------------------------

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(input_shape)))
model.add(MaxPooling2D(pool_size=(2), strides=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2), strides=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=6, activation='softmax'))

model.summary()
# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 224, 224, 32)      896
#
#  max_pooling2d (MaxPooling2D  (None, 112, 112, 32)     0
#  )
#
#  conv2d_1 (Conv2D)           (None, 112, 112, 64)      18496
#
#  max_pooling2d_1 (MaxPooling  (None, 56, 56, 64)       0
#  2D)
#
#  conv2d_2 (Conv2D)           (None, 56, 56, 32)        18464
#
#  max_pooling2d_2 (MaxPooling  (None, 28, 28, 32)       0
#  2D)
#
#  flatten (Flatten)           (None, 25088)             0
#
#  dense (Dense)               (None, 64)                1605696
#
#  dropout (Dropout)           (None, 64)                0
#
#  dense_1 (Dense)             (None, 32)                2080
#
#  dropout_1 (Dropout)         (None, 32)                0
#
#  dense_2 (Dense)             (None, 6)                 198
#
# =================================================================
# Total params: 1,645,830
# Trainable params: 1,645,830
# Non-trainable params: 0
# _________________________________________________________________


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'accuracy'])


callbacks =[EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min'),
            ModelCheckpoint(filepath='CNN_model.h5', monitor='val_loss', mode='min', save_best_only=True,
                            save_weights_only=False, verbose=1)]

history = model.fit_generator(generator=train_generator,
                              epochs=100,
                              validation_data=test_generator,
                              callbacks=callbacks,
                              workers=4,
                              steps_per_epoch=2276//32,
                              validation_steps=251//32)


# -------------------------------------------------
# 5. Model Evaluation
# -------------------------------------------------

plt.figure(figsize=(20, 5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], color='blue', label='Training accuracy')
plt.plot(history.history['val_accuracy'], color='red', label='Validation accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and validation accuracy', fontsize=20)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], color='blue', label='Training loss')
plt.plot(history.history['val_loss'], color='red', label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and validation loss', fontsize=20)
plt.show()


# Metrics:

loss, precision, recall, accuracy = model.evaluate(test_generator, batch_size=32)

print(f'Test accuracy: {accuracy*100:.2f}%')
print(f'Test precision: {precision*100:.2f}%')
print(f'Test recall: {recall*100:.2f}%')
print(f'Test loss: {loss*100:.2f}%')

# Test accuracy: 71.31%
# Test precision: 79.33%
# Test recall: 65.74%
# Test loss: 81.88%


# Classification Report:

x_test, y_test = test_generator.next()
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

target_names = list(waste_labels.keys())

print(classification_report(y_test, y_pred, target_names=target_names))

#               precision    recall  f1-score   support
#    cardboard       0.74      0.62      0.68        40
#        glass       0.83      0.70      0.76        50
#        metal       0.52      0.76      0.61        41
#        paper       0.83      0.97      0.89        59
#      plastic       0.70      0.44      0.54        48
#        trash       0.62      0.77      0.69        13
#     accuracy                           0.71       251
#    macro avg       0.71      0.71      0.69       251
# weighted avg       0.73      0.71      0.71       251

# Confusion Matrix:

cm = confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')
    plt.show()

plot_confusion_matrix(cm, waste_labels.keys())















