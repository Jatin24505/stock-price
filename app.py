# app.py
from flask import Flask, render_template, request
import yfinance as yf
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
import os

app = Flask(__name__)

def predict_and_plot(ticker):
    df = yf.download(ticker, start='2015-01-01', end='2023-01-01')
    data = df[['Close']]
    scaler = joblib.load(f'{ticker}_scaler.save')
    scaled_data = scaler.transform(data)

    sequence_length = 60
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    x = np.array(x)
    y = np.array(y)

    model = load_model(f'{ticker}_model.h5')
    predictions = model.predict(x)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y)

    plt.figure(figsize=(10,6))
    plt.plot(actual, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        try:
            if not os.path.exists(f'{ticker}_model.h5'):
                from model import train_model
                train_model(ticker)
            plot_path = predict_and_plot(ticker)
            return render_template('index.html', plot_url=plot_path, ticker=ticker)
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
