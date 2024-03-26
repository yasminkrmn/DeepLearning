#-----------------------------------------------
# AIRLINE PASSENGER NUMBERS FORECAST
#-----------------------------------------------
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#-----------------------------------------------
# 1. Importing & Understanding Data
#-----------------------------------------------

df = pd.read_csv('3_RNN/international-airline-passengers.csv')
df.head()

#      Month  International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60
# 0  1949-01                                              112.0
# 1  1949-02                                              118.0
# 2  1949-03                                              132.0
# 3  1949-04                                              129.0
# 4  1949-05                                              121.0

df.columns = ['Month', 'Passengers']
df.head()
#      Month  Passengers
# 0  1949-01       112.0
# 1  1949-02       118.0
# 2  1949-03       132.0
# 3  1949-04       129.0
# 4  1949-05       121.0

df.tail()
#                                                  Month  Passengers
# 140                                            1960-09       508.0
# 141                                            1960-10       461.0
# 142                                            1960-11       390.0
# 143                                            1960-12       432.0
# 144  International airline passengers: monthly tota...         NaN

df.shape
# (145, 2)

df.info()
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   Month       145 non-null    object
#  1   Passengers  144 non-null    float64
# dtypes: float64(1), object(1)

df.describe().T
#             count        mean         std    min    25%    50%    75%    max
# Passengers  144.0  280.298611  119.966317  104.0  180.0  265.5  360.5  622.0


#-----------------------------------------------
# 2. Data Preparation
#-----------------------------------------------

#--------------- Dropping Na Values ---------------------

df = df.dropna(axis=0)
df.tail()
#        Month  Passengers
# 139  1960-08       606.0
# 140  1960-09       508.0
# 141  1960-10       461.0
# 142  1960-11       390.0
# 143  1960-12       432.0

#-------------- Changing TO Datetime ------------------
df.dtypes
# Month          object
# Passengers    float64

df['Month'] = pd.to_datetime(df['Month'])

df['Month'].min(), df['Month'].max()
# (Timestamp('1949-01-01 00:00:00'), Timestamp('1960-12-01 00:00:00'))

#------------------ Changing Index -------------------------
df.set_index('Month', inplace=True)

df.head()
#             Passengers
# Month
# 1949-01-01       112.0
# 1949-02-01       118.0
# 1949-03-01       132.0
# 1949-04-01       129.0
# 1949-05-01       121.0

#------------------ Visualizing Data -------------------------

df.plot(figsize=(10, 10))
plt.title('Monthly Airline Passengers')
plt.xlabel('Month')
plt.ylabel('Count in Thousand of Passengers')
plt.legend(loc='best')
plt.show()

#------------------ Changing TO NDARRAY -------------------------
data = df['Passengers'].values
data[0:5]
data = data.reshape(-1, 1).astype('float32')
data[0:5]
# array([[112.],
#        [118.],
#        [132.],
#        [129.],
#        [121.]], dtype=float32)

#------------------ Train - Test Splitting ----------------------

def train_test_split(data, test_size):
    position = int(round(len(data) * (1-test_size)))
    train = data[:position]
    test = data[position:]
    return train, test, position
train, test, position = train_test_split(data, test_size=0.33)
print(f'Train Size: {len(train)} -', f'Test Size: {len(test)} -', f'Position: {position}')
# Train Size: 96 - Test Size: 48 - Position: 96

print(train.shape, test.shape)
# (96, 1) (48, 1)

#----------------------- MinMax Scaling -------------------------

scaler_train = MinMaxScaler(feature_range=(0,1))
train = scaler_train.fit_transform(train)
train[0:3]
# array([[0.02588999],
#        [0.04530746],
#        [0.09061491]], dtype=float32)

scaler_test = MinMaxScaler(feature_range=(0,1))
test = scaler_test.fit_transform(test)
test[0:3]
# array([[0.04361373],
#        [0.        ],
#        [0.17133951]], dtype=float32)

def create_features(data, lookback):
    """
    Create input features and target labels from time series data.

    Parameters:
    - data (numpy.ndarray): The input time series data.
    - lookback (int): The number of previous time steps to use as features.

    Returns:
    - X (numpy.ndarray): Input features, a 2D numpy array where each row contains the
      values of the previous 'lookback' time steps.
    - y (numpy.ndarray): Target labels, a 1D numpy array where each element represents
      the value of the time step immediately following the corresponding row in X.
    """
    X = []
    y = []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i,0])
        y.append(data[i,0])

    return np.array(X), np.array(y)


X_train, y_train = create_features(train, lookback=1)
X_test, y_test = create_features(test, lookback=1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# (95, 1) (95,) (47, 1) (47,)
"""
type(X_train.reshape(95,1,1).shape)
X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
"""

X_train = X_train.reshape(X_train.shape[0],1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1, X_test.shape[1])
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

#-----------------------------------------------
# 3. Modeling
#-----------------------------------------------
model = Sequential()
model.add(SimpleRNN(
    units=50,
    activation='relu',
    input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 50)                2600
#
#  dropout (Dropout)           (None, 50)                0
#
#  dense (Dense)               (None, 1)                 51
#
# =================================================================
# Total params: 2,651
# Trainable params: 2,651
# Non-trainable params: 0
# _________________________________________________________________

model.compile(loss='mean_squared_error', optimizer='adam')
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min'),
             ModelCheckpoint(filepath='SimpleRNN.h5', monitor='val_loss', mode='min',
                             save_best_only=True, save_weights_only=False, verbose=1)]


history = model.fit(x=X_train, y=y_train,
                    epochs=50, batch_size=1,
                    callbacks=callbacks, validation_data=(X_test, y_test),
                    shuffle=False)



# -------------------------------------------------
# 5. Model Evaluation
# -------------------------------------------------

# LOSS:

plt.figure(figsize=(10, 5))
plt.subplot(1,2,2)
plt.plot(history.history['loss'], color='blue', label='Training loss')
plt.plot(history.history['val_loss'], color='red', label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and validation loss', fontsize=20)
plt.show()


loss = model.evaluate(X_test, y_test, batch_size=1)
print('\nTest loss: %.1f%%' % (100*loss))
# Test loss: 2.0%

# RMSE:
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler_train.inverse_transform(train_predict)
test_predict = scaler_test.inverse_transform(test_predict)

y_train = scaler_train.inverse_transform(y_train)
y_test = scaler_test.inverse_transform(y_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

print('Train RMSE:', np.round(train_rmse,2))
print('Test RMSE:', np.round(test_rmse,2))

# Train RMSE: 23.62
# Test RMSE: 45.87


print('Test Mean:', np.round(np.mean(y_test),2))
print('Test Std:', np.round(np.std(y_test),2))

# Test Mean: 415.57
# Test Std: 77.14














