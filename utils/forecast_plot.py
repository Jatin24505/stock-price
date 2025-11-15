
import pandas as pd

import plotly.graph_objs as go
from plotly.offline import plot

def generate_plot(actual_series, forecast_values, forecast_start_date):
    # Dates for forecast
    forecast_dates = [forecast_start_date + pd.Timedelta(days=i) for i in range(len(forecast_values))]

    # Create traces
    actual_trace = go.Scatter(
        x=actual_series.index,
        y=actual_series.values,
        mode='lines',
        name='Actual Price'
    )

    forecast_trace = go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines',
        name='Forecasted Price'
    )

    layout = go.Layout(
        title='Predicted vs Actual Stock Prices',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price (USD)'),
        legend=dict(x=0, y=1),
        hovermode='x unified'
    )

    fig = go.Figure(data=[actual_trace, forecast_trace], layout=layout)
    return plot(fig, output_type='div')

