import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
class StockPricePredictor:
    def __init__(self, model, ticker, start_date, end_date, window_size=20):
        self.model = model
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.scaler_train = MinMaxScaler(feature_range=(0, 1))
        self.scaler_test = MinMaxScaler(feature_range=(0, 1))


    def predict_price(self, real_df):
        real_df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
        real_df = real_df.values.astype('float32')

        input_data = self.scaler_test.fit_transform(real_df)
        input_data = input_data.reshape(1, 1, self.window_size)

        self.model = load_model(model)

        predicted_price = self.model.predict(input_data)
        predicted_price = self.scaler_test.inverse_transform(predicted_price)

        return predicted_price


if __name__ == '__main__':
    ticker = 'TSLA'
    start_date = '2024-02-27'
    end_date = '2024-03-26'
    model= 'LSTM.h5'

    stock_predictor = StockPricePredictor(model = model, ticker=ticker, start_date=start_date, end_date=end_date)
    actual_data = yf.download(ticker, start=start_date, end=end_date)
    predicted_price = stock_predictor.predict_price(actual_data)

    end_date_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    previous_date = end_date_datetime - timedelta(days=1)

    print(f'Actual Close Price of TSLA on {previous_date.strftime("%Y-%m-%d")}: {float(actual_data.Close[-1]):.2f}$')
    print(f'Predicted Close Price of TSLA on {previous_date.strftime("%Y-%m-%d")}: {float(predicted_price):.2f}$')

# Actual Close Price of TSLA on 2024-03-25: 172.63$
# Predicted Close Price of TSLA on 2024-03-25: 178.57$