import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def show_regression_analysis(regr_data, asset_type, config):
    """
    Perform linear regression forecasting on price data and visualize the forecast.

    :param regr_data: DataFrame containing historical asset price data with a 'CLOSE' column
    :param asset_type: Type of asset being analyzed (e.g., "Stock", "Metal")
    :param config: Configuration dictionary with forecast defaults
    :return: Tuple containing (Plotly Figure, forecast array, R² score, forecast date range)
    """
    st.subheader(f"🔮 {asset_type} Price Forecast with Linear Regression")

    default_days = config["forecast_defaults"].get(asset_type, {}).get("regression_days", 8)

    forecast_days = st.slider(
        "Days to forecast:",
        min_value=7,
        max_value=10,
        value=default_days,
        key=f"regression_{asset_type.lower()}"
    )

    if regr_data is None or regr_data.empty:
        st.warning(f"No {asset_type.lower()} data available for regression.")
        return None, None, None, None

    full_df = regr_data[['CLOSE']].dropna().copy()
    full_df['Days'] = np.arange(len(full_df))

    plot_df = full_df[-255:] if len(full_df) >= 255 else full_df.copy()

    X = full_df[['Days']]
    y = full_df['CLOSE']

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(len(full_df), len(full_df) + forecast_days)
    future_X = pd.DataFrame(future_days, columns=['Days'])
    forecast = model.predict(future_X)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    forecast_dates = pd.date_range(start=full_df.index[-1] + pd.Timedelta(days=1),
                                   periods=forecast_days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index.to_list(),
        y=plot_df['CLOSE'].values,
        name="Actual (Last Year)",
        line=dict(color='cyan')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates.to_list(),
        y=forecast,
        name="Forecast",
        line=dict(color='magenta', dash='dot')
    ))

    start_date = plot_df.index[0] - pd.Timedelta(days=3)
    end_date = forecast_dates[-1] + pd.Timedelta(days=3)

    fig.update_layout(
        template="plotly_dark",
        title=f"{asset_type} Regression Price Forecast",
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

    st.info(f"""
    **Model Performance Metrics (Full History):**  
    R² score: {r2:.4f}  
    MAE: {mae:.2f}  
    RMSE: {rmse:.2f}
    """)

    return fig, forecast, r2, forecast_dates
