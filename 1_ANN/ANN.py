# -------------------------------------------------
# PREDICTION OF HANDWRITTEN DIGITS - MNIST
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

# -------------------------------------------------
# 3. Data Preparation
# -------------------------------------------------

# 3.1 Encoding:

# Before:
np.unique(y_train)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)

# After:
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
np.unique(y_train)
# array([0., 1.], dtype=float32)
y_train[0]
# array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=float32)



# 3.2 Reshaping:

# Before:
x_train.shape, x_test.shape
# (60000, 28, 28),  (10000, 28, 28)

# After:
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train.shape, x_test.shape
# (60000, 28, 28, 1), (10000, 28, 28, 1)

# 3.3 Normalization:

# The operation scales the pixel values of the data to a range between 0 and 1.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# -------------------------------------------------
# 4. Modeling
# -------------------------------------------------

model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu', name='hidden_layer_1'),
    Dense(10, activation='softmax', name='output_layer')
])

model.compile('adam', 'categorical_crossentropy',
              metrics=[tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.AUC(),
                       'accuracy'])

model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  flatten_1 (Flatten)         (None, 784)               0          (28*28) = 784 pixels
#
#  hidden_layer_1 (Dense)      (None, 128)               100480     (784*128) + 128 Bias = 100400
#
#  output_layer (Dense)        (None, 10)                1290       (128*10) + 10 Bias = 1290
#
# =================================================================
# Total params: 101,770
# Trainable params: 101,770
# Non-trainable params: 0
# _________________________________________________________________

model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))


# -------------------------------------------------
# 5. Model Evaluation
# -------------------------------------------------

history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

plt.figure(figsize=(20,5))
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
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and validation loss', fontsize=20)
plt.show()

loss, precision, recall, auc, accuracy = model.evaluate(x_test, y_test, verbose=False)
print(f'Test accuracy: {accuracy*100:.2f}%')
print(f'Test precision: {precision*100:.2f}%')
print(f'Test recall: {recall*100:.2f}%')
print(f'Test auc: {auc*100:.2f}%')
print(f'Test loss: {loss*100:.2f}%')

# Test accuracy: 96.08%
# Test precision: 96.32%
# Test recall: 95.91%
# Test auc: 99.18%
# Test loss: 25.26%

# -------------------------------------------------
# 6. Model Saving and Prediction
# -------------------------------------------------

model.save('mnistANNmodel.h5')

import random

random_image = random.randint(0, x_test.shape[0])
random_image
# 4972
prediction = model.predict(x_test[random_image].reshape(1, 28, 28, 1))

prediction
# array([[1.0546625e-15, 9.9880111e-01, 1.3662964e-07, 4.6622669e-14,
#         1.6926519e-08, 1.1963803e-11, 5.4908159e-16, 1.1987607e-03,
#         3.0164393e-10, 9.9095289e-12]], dtype=float32)

np.argmax(prediction)
# 1

y_test[random_image]
# array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)

plt.imshow(x_test[random_image], cmap='gray')
plt.show()
# Shows: 1

print(f'Predicted label: {np.argmax(prediction)}')
print(f'Actual label: {y_test[random_image]}')
print(f'Predicted probability: {np.round(np.max(prediction, axis=-1)[0], 2)}')

# Predicted label: 1
# Actual label: [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# Predicted probability: 1.0










