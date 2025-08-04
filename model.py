# model.py
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import joblib

def train_model(ticker='AAPL'):
    df = yf.download(ticker, start='2015-01-01', end='2023-01-01')
    df = df[['Close']]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    sequence_length = 60
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    x, y = np.array(x), np.array(y)

    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    model.save(f'{ticker}_model.h5')
    joblib.dump(scaler, f'{ticker}_scaler.save')
    return f'{ticker}_model.h5', f'{ticker}_scaler.save'
