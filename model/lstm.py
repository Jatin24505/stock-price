import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_and_forecast(df, days=15):
    """
    Train LSTM model and forecast future prices.
    
    Args:
        df: DataFrame with 'Date' and 'Close' columns
        days: Number of days to forecast
    
    Returns:
        DataFrame with 'Date' and 'Forecast' columns
    """
    # Make a copy and ensure Date is datetime
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Extract the Close prices as a series
    series = df["Close"].values

    # Scale data to [0, 1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))

    look_back = 60
    
    # Need at least look_back + 1 data points
    if len(scaled) < look_back + 1:
        raise ValueError(f"Need at least {look_back + 1} data points, got {len(scaled)}")
    
    X, y = [], []

    # Create sequences
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])

    if len(X) == 0:
        raise ValueError("No training sequences created")

    X = np.array(X).reshape((-1, look_back, 1))
    y = np.array(y)

    # LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=8, batch_size=32, verbose=0)

    # Forecasting
    input_seq = scaled[-look_back:]
    forecast_scaled = []

    for _ in range(days):
        pred = model.predict(input_seq.reshape(1, look_back, 1), verbose=0)[0][0]
        forecast_scaled.append(pred)
        input_seq = np.append(input_seq[1:], pred)

    # Inverse scale
    forecast = scaler.inverse_transform(
        np.array(forecast_scaled).reshape(-1, 1)
    ).flatten()

    # Generate future dates
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)

    return pd.DataFrame({
        "Date": forecast_dates,
        "Forecast": forecast
    })