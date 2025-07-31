import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def show_lstm_analysis(lstm_data, asset_type, config):
    """
    Display an LSTM-based price forecast for a given asset type using historical data.

    :param lstm_data: Pandas DataFrame containing historical price data with a 'CLOSE' column.
    :param asset_type: String representing the type of asset (e.g., "Stock", "Metal").
    :param config: Configuration dictionary containing forecast defaults.
    :return: Tuple containing:
        - Plotly figure of historical and forecasted prices,
        - Numpy array of forecasted prices,
        - Pandas DatetimeIndex of forecast dates,
        - Plotly figure of training loss.
        Returns (None, None, None, None) on error or insufficient data.
    """
    try:
        if lstm_data is None or lstm_data.empty:
            st.warning(f"No {asset_type.lower()} data available for LSTM.")
            return

        default_days = config["forecast_defaults"].get(asset_type, {}).get("lstm_days", 20)

        st.markdown(f"#### 🤖 {asset_type} Price Forecast (LSTM)")

        lstm_days = st.slider(
            "Days to forecast:",
            min_value=14,
            max_value=21,
            value=default_days,
            key=f"lstm_{asset_type.lower()}"
        )

        df = lstm_data[['CLOSE']].dropna()
        if len(df) < 50:
            st.warning(f"Not enough data points ({len(df)}). Need at least 50 for LSTM.")
            return

        data_values = df.values  # Last 300 days for model training

        plot_data = df[-255:] if len(df) >= 255 else df

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_values)

        window_size = 20
        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i - window_size:i])
            y.append(scaled_data[i])
        X, y = np.array(X), np.array(y)

        if len(X) == 0 or len(y) == 0:
            st.warning("Not enough data to create training windows for LSTM.")
            return None, None, None, None

        tf.keras.backend.clear_session()

        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')

        validation_split_ratio = 0.1 if len(X) > 20 else 0

        with st.spinner(f"Training LSTM model for {asset_type}..."):
            history = model.fit(
                X,
                y,
                epochs=20,
                batch_size=16,
                verbose=0,
                validation_split=validation_split_ratio,
                shuffle=False
            )

        last_window = scaled_data[-window_size:].reshape(1, window_size, 1)
        forecast_scaled = []

        for _ in range(lstm_days):
            pred = model.predict(last_window, verbose=0)
            forecast_scaled.append(pred[0, 0])
            last_window = np.append(last_window[:, 1:, :], [[[pred[0, 0]]]], axis=1)

        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

        forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=lstm_days)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_data.index.to_list(),
            y=plot_data['CLOSE'].values,
            name="Historical",
            line=dict(color='lightblue')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates.to_list(),
            y=forecast,
            name="LSTM Forecast",
            line=dict(color='orange', dash='dot')
        ))

        start_date = plot_data.index[0] - pd.Timedelta(days=3)
        end_date = forecast_dates[-1] + pd.Timedelta(days=3)

        fig.update_layout(
            template="plotly_dark",
            title=f"{asset_type} LSTM Price Forecast",
            xaxis=dict(
                title="Date",
                type="date",
                tickformat="%b %Y",
                hoverformat="%b %d, %Y",
                range=[start_date, end_date]
            ),
            yaxis_title="Price",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=history.history['loss'],
            name='Training Loss',
            line=dict(color='red')
        ))
        if 'val_loss' in history.history:
            fig_loss.add_trace(go.Scatter(
                y=history.history['val_loss'],
                name='Validation Loss',
                line=dict(color='blue')
            ))
        fig_loss.update_layout(
            title="Model Training Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=300
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        return fig, forecast, forecast_dates, fig_loss

    except Exception as e:
        st.error(f"Error in LSTM analysis: {str(e)}")
        return None, None, None, None
