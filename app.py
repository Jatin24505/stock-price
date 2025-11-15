from flask import Flask, render_template, request, send_file
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import os
from model.lstm import train_and_forecast
import yfinance as yf
import numpy as np

app = Flask(__name__)

STATIC_FOLDER = "static"
PLOT_PATH = os.path.join(STATIC_FOLDER, "plot.png")
CSV_PATH = os.path.join(STATIC_FOLDER, "forecast.csv")


@app.route("/", methods=["GET", "POST"])
def home():
    ticker = "AAPL"
    days = 15
    fig_html = None

    last_close_price = None
    last_month_avg = None
    prev_month_avg = None
    growth_pct = None
    forecast_method = "LSTM"

    if request.method == "POST":
        ticker = request.form.get("ticker", "AAPL")
        days = int(request.form.get("days", 15))

        # -------------------------  
        # FETCH STOCK DATA  
        # -------------------------
        print(f"Fetching data for {ticker}...")
        df = yf.download(ticker, period="3mo")
        
        print(f"Downloaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")

        if df.empty:
            return render_template("index.html", error="No data found for this ticker.")

        df = df.reset_index()
        
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[["Date", "Close"]]

        # Basic values
        last_close_price = df["Close"].iloc[-1]

        # Handle small datasets
        last_month = df.tail(30)
        prev_month = df.tail(60).head(30)

        last_month_avg = last_month["Close"].mean() if len(last_month) > 0 else None
        prev_month_avg = prev_month["Close"].mean() if len(prev_month) > 0 else None

        # -------------------------  
        # LSTM FORECAST  
        # -------------------------
        try:
            # Pass the full DataFrame (not just the Series)
            print(f"Starting LSTM forecast for {days} days...")
            forecast_df = train_and_forecast(df, days)
            print(f"Forecast generated: {len(forecast_df)} rows")
            print(forecast_df.head())
            forecast_method = "LSTM"
        except Exception as e:
            print(f"LSTM failed: {e}")
            # Fallback to linear trend
            forecast_method = "Linear Trend (LSTM failed)"
            last_date = pd.to_datetime(df["Date"].iloc[-1])
            last_price = float(df["Close"].iloc[-1])
            
            # Simple linear trend
            recent = df.tail(30)
            if len(recent) > 1:
                slope = (float(recent["Close"].iloc[-1]) - float(recent["Close"].iloc[0])) / len(recent)
            else:
                slope = 0
            
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            forecast_values = [last_price + slope * (i + 1) for i in range(days)]
            
            forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                "Forecast": forecast_values
            })
            print(f"Fallback forecast generated: {len(forecast_df)} rows")

        # Safe growth % (comparing last close to first forecast value)
        try:
            growth_pct = ((float(forecast_df["Forecast"].iloc[0]) - float(last_close_price)) / float(last_close_price)) * 100
        except:
            growth_pct = None

        # -------------------------  
        # CONVERT ALL VALUES TO FLOATS  
        # -------------------------
        def safe_float(x):
            try:
                return float(x)
            except:
                return None

        last_close_price = safe_float(last_close_price)
        last_month_avg = safe_float(last_month_avg)
        prev_month_avg = safe_float(prev_month_avg)
        growth_pct = safe_float(growth_pct)

        # -------------------------  
        # BUILD LINE CHART  
        # -------------------------
        print(f"Building chart...")
        print(f"Previous month data: {len(prev_month)} rows")
        print(f"Last month data: {len(last_month)} rows")
        print(f"Forecast data: {len(forecast_df)} rows")
        
        fig = go.Figure()

        # Historical data - Previous Month (Blue line)
        if len(prev_month) > 0:
            fig.add_trace(go.Scatter(
                x=prev_month["Date"],
                y=prev_month["Close"],
                mode="lines",
                name="Previous Month",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="<b>Previous Month</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
            ))

        # Historical data - Last Month (Red line)
        if len(last_month) > 0:
            fig.add_trace(go.Scatter(
                x=last_month["Date"],
                y=last_month["Close"],
                mode="lines",
                name="Last Month",
                line=dict(color="#ff7f0e", width=2),
                hovertemplate="<b>Last Month</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
            ))

        # Forecast - Dashed Green line
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast"],
            mode="lines",
            name="Forecast (LSTM)",
            line=dict(color="#2ca02c", dash="dash", width=3),
            hovertemplate="<b>Forecast</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
        ))

        # Update layout for better line chart visualization
        fig.update_layout(
            title={
                'text': f"{ticker} Stock Price Forecast - Line Chart",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#333'}
            },
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=550,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
        )

        fig_html = pio.to_html(fig, full_html=False)

        # Save PNG & CSV
        os.makedirs(STATIC_FOLDER, exist_ok=True)
        fig.write_image(PLOT_PATH)
        forecast_df.to_csv(CSV_PATH, index=False)

    return render_template(
        "index.html",
        plot_div=fig_html,
        ticker=ticker,
        days=days,
        last_close_price=last_close_price,
        last_month_avg=last_month_avg,
        prev_month_avg=prev_month_avg,
        growth_pct=growth_pct,
        forecast_method=forecast_method
    )


@app.route("/download_png")
def download_png():
    return send_file(PLOT_PATH, as_attachment=True)


@app.route("/download_csv")
def download_csv():
    return send_file(CSV_PATH, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)