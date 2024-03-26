#-----------------------------------------------
# STOCK PRICE PREDICTION
#-----------------------------------------------
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#-----------------------------------------------
# 1. Importing & Understanding Data
#-----------------------------------------------
ticker = 'TSLA'
start_date = '2010-01-01'
end_date = '2024-03-01'

df = yf.download(ticker, start=start_date, end=end_date)

print(df.head())

def data_summary(dataframe):

    # Displaying head of the DataFrame
    print("----------------------------------------------")
    print("Head of the DataFrame:")
    print(dataframe.head())
    print("----------------------------------------------")

    # Displaying tail of the DataFrame
    print("Tail of the DataFrame:")
    print(dataframe.tail())
    print("----------------------------------------------")

    # Displaying shape of the DataFrame
    print("Shape of the DataFrame:")
    print(dataframe.shape)
    print("----------------------------------------------")

    # Displaying information about the DataFrame
    print("Information about the DataFrame:")
    print(dataframe.info())
    print("----------------------------------------------")

    # Calculating total number of missing values
    print("Missing values:")
    print(dataframe.isnull().sum())
    print("----------------------------------------------")

    # Displaying descriptive statistics
    print("Descriptive statistics:")
    print(dataframe.describe().T)
    print("----------------------------------------------")

data_summary(df)
# ----------------------------------------------
# Head of the DataFrame:
#                 Open      High       Low     Close  Adj Close     Volume
# Date
# 2010-06-29  1.266667  1.666667  1.169333  1.592667   1.592667  281494500
# 2010-06-30  1.719333  2.028000  1.553333  1.588667   1.588667  257806500
# 2010-07-01  1.666667  1.728000  1.351333  1.464000   1.464000  123282000
# 2010-07-02  1.533333  1.540000  1.247333  1.280000   1.280000   77097000
# 2010-07-06  1.333333  1.333333  1.055333  1.074000   1.074000  103003500
# ----------------------------------------------
# Tail of the DataFrame:
#                   Open        High  ...   Adj Close     Volume
# Date                                ...
# 2024-03-18  170.020004  174.720001  ...  173.800003  108214400
# 2024-03-19  172.360001  172.820007  ...  171.320007   77271400
# 2024-03-20  173.000000  176.250000  ...  175.660004   83846700
# 2024-03-21  176.389999  178.179993  ...  172.820007   73178000
# 2024-03-22  166.690002  171.199997  ...  170.830002   75454700
# [5 rows x 6 columns]
# ----------------------------------------------
# Shape of the DataFrame:
# (3457, 6)
# ----------------------------------------------
# Information about the DataFrame:
# <class 'pandas.core.frame.DataFrame'>
# DatetimeIndex: 3457 entries, 2010-06-29 to 2024-03-22
# Data columns (total 6 columns):
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   Open       3457 non-null   float64
#  1   High       3457 non-null   float64
#  2   Low        3457 non-null   float64
#  3   Close      3457 non-null   float64
#  4   Adj Close  3457 non-null   float64
#  5   Volume     3457 non-null   int64
# dtypes: float64(5), int64(1)
# memory usage: 189.1 KB
# None
# ----------------------------------------------
# Missing values:
# Open         0
# High         0
# Low          0
# Close        0
# Adj Close    0
# Volume       0
# dtype: int64
# ----------------------------------------------
# Descriptive statistics:
#             count          mean  ...           75%           max
# Open       3457.0  7.258163e+01  ...  1.369500e+02  4.114700e+02
# High       3457.0  7.416460e+01  ...  1.409633e+02  4.144967e+02
# Low        3457.0  7.086982e+01  ...  1.350033e+02  4.056667e+02
# Close      3457.0  7.255550e+01  ...  1.372533e+02  4.099700e+02
# Adj Close  3457.0  7.255550e+01  ...  1.372533e+02  4.099700e+02
# Volume     3457.0  9.692664e+07  ...  1.230780e+08  9.140820e+08
# [6 rows x 8 columns]
# ----------------------------------------------


#------------------ Selecting Close Price -------------------------
data = df[['Close']]
data.head()
#                Close
# Date
# 2010-06-29  1.592667
# 2010-06-30  1.588667
# 2010-07-01  1.464000
# 2010-07-02  1.280000
# 2010-07-06  1.074000


#------------------ Visualizing Data -------------------------
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label='Close', color='blue')
plt.title('Tesla Close Price')
plt.ylabel('Close Price')
plt.xlabel('Date')
plt.legend()
plt.show()

#-----------------------------------------------
# 2. Data Preparation
#-----------------------------------------------

#------------------ Changing TO NDARRAY -------------------------
tesla_df = data.copy()

tesla_df = tesla_df.values.astype('float32')
tesla_df[0:5]
# array([[1.59266698],
#        [1.58866704],
#        [1.46399999],
#        [1.27999997],
#        [1.074     ]])

tesla_df.shape
# (3457, 1)

#------------------ Train - Test Splitting ----------------------

def split_data(data, test_size):
    position = int(round(len(data) * (1-test_size)))
    train_data = data[:position]
    test_data = data[position:]
    return train_data, test_data, position


train_data, test_data, position = split_data(tesla_df, test_size=0.20)

print(train_data.shape, test_data.shape, position)
# (2766, 1) (691, 1) 2766

#----------------------- MinMax Scaling -------------------------

scaler_train = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler_train.fit_transform(train_data)
scaler_test = MinMaxScaler(feature_range=(0,1))
scaled_test = scaler_test.fit_transform(test_data)

scaled_train[0:5], scaled_test[0:5]
# (array([[1.8387847e-03],
#         [1.8251473e-03],
#         [1.4001122e-03],
#         [7.7278959e-04],
#         [7.0461072e-05]], dtype=float32),
#  array([[0.3925752 ],
#         [0.38379657],
#         [0.40240282],
#         [0.39361316],
#         [0.39244264]], dtype=float32))


#---------------- Extracting Sequantial X and Y -------------------
def extract_seqX_Y(data, window_size):
    """
    Extract sequence input features X and target labels Y from time series data.

    Parameters:
    - data (numpy.ndarray): The input time series data.
    - window size, e.g., 20 for 20 days of historical stock prices

    Returns:
    - X (numpy.ndarray): Input features, a 2D numpy array where each row contains the
      values of the previous 'window_size' time steps.
    - y (numpy.ndarray): Target labels, a 1D numpy array where each element represents
      the value of the time step immediately following the corresponding row in X.
    """
    X = []
    y = []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i,0])
        y.append(data[i,0])

    return np.array(X), np.array(y)

window_size = 20


X_train, y_train = extract_seqX_Y(scaled_train, window_size)
X_test, y_test = extract_seqX_Y(scaled_test, window_size)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# (2746, 20) (2746,) (671, 20) (671,)


#---------------- 2-Dimension To 3-Dimension -------------------

X_train = X_train.reshape(X_train.shape[0],1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1, X_test.shape[1])
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# (2746, 1, 20) (2746, 1) (671, 1, 20) (671, 1)

#-----------------------------------------------
# 3. Modeling
#-----------------------------------------------

model = Sequential()
model.add(LSTM(units=50,
               activation='relu',
               input_shape= (X_train.shape[1], 20)))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 50)                14200
#
#  dropout (Dropout)           (None, 50)                0
#
#  dense (Dense)               (None, 1)                 51
#
# =================================================================
# Total params: 14,251
# Trainable params: 14,251
# Non-trainable params: 0
# _________________________________________________________________


model.compile(loss='MeanSquaredError', optimizer='adam')
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min'),
             ModelCheckpoint(filepath='LSTM.h5', monitor='val_loss', mode='min',
                             save_best_only=True, save_weights_only=False, verbose=1)]

history = model.fit(X_train, y_train,
                    epochs=50, batch_size=20,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    shuffle=False)


# -------------------------------------------------
# 4. Model Evaluation
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
plt.show()



loss = model.evaluate(X_test, y_test, batch_size=1)
print('\nTest loss: %.1f%%' % (100*loss))
# Test loss: 1.4%

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

# Train RMSE: 17.47
# Test RMSE: 35.47

print('Test Mean:', np.round(np.mean(y_test),2))
print('Test Std:', np.round(np.std(y_test),2))

# Test Mean: 245.68
# Test Std: 59.16

# -------------------------------------------------
# 5. Visualization of Prediction
# -------------------------------------------------

train_predicted_df = data[20:position]
train_predicted_df["Predicted"] = train_predict
train_predicted_df.head(3)
#                Close  Predicted
# Date
# 2010-07-28  1.381333   7.732932
# 2010-07-29  1.356667   7.732932
# 2010-07-30  1.329333   7.732932

test_predicted_df = data[20+position:]
test_predicted_df["Predicted"] = test_predict
test_predicted_df.tail(3)
#                  Close   Predicted
# Date
# 2024-02-27  199.729996  174.875977
# 2024-02-28  202.039993  175.784164
# 2024-02-29  201.880005  175.575851

plt.figure(figsize=(14,5))
plt.plot(data, label='Actual Price')
plt.plot(train_predicted_df['Predicted'], label='Train Predicted Price', color='blue')
plt.plot(test_predicted_df['Predicted'], label='Test Predicted Price', color='red')
plt.legend(loc='upper left')
plt.xlabel('Time')
plt.ylabel('Predicted Price')
plt.show()





# Dünkü kapanış fiyatını alın

df.Close = 172.63
yesterday_close_price = df['Close'].iloc[-1]

# Modelin beklendiği forma dönüştürün
input_data = np.array([[yesterday_close_price]])
input_data = scaler_test.transform(input_data)  # Veriyi ölçeklendirme
input_data = input_data.reshape(1, 1, 1)  # Giriş verisini modelin beklediği forma dönüştürme

# Tahmini yapın
today_price_prediction = model.predict(input_data)

# Tahmin edilen fiyatı geri ölçeklendirin
today_price_prediction = scaler_test.inverse_transform(today_price_prediction)

print("Tahmin edilen bugünkü kapanış fiyatı:", today_price_prediction)






#-----------------------------------------------
# 1. Importing & Understanding Data
#-----------------------------------------------
ticker = 'TSLA'
start_date = '2024-01-01'
end_date = '2024-03-26'

tesla_data = yf.download(ticker, start=start_date, end=end_date)

print(tesla_data.tail())







