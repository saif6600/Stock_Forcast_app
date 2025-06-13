import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Page Setup
st.set_page_config(page_title="Stock Forecast", layout="wide")
st.title("📈 Stock Price Forecasting")
st.markdown("Upload your stock CSV file to see forecasts using ARIMA, SARIMA, and Prophet.")

# File Upload
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    # Load and preprocess
    df = pd.read_csv(file, skiprows=2)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(inplace=True)
    df.set_index('Date', inplace=True)

    st.line_chart(df['Close'])

    forecast_days = st.slider("Forecast Days", 7, 90, 30)

    # ARIMA Forecast
    st.subheader("🔮 ARIMA Forecast")
    try:
        arima_model = ARIMA(df['Close'], order=(5, 1, 0))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=forecast_days)
        st.line_chart(pd.concat([df['Close'], arima_forecast.rename("Forecast")]))
    except Exception as e:
        st.error(f"ARIMA Error: {e}")

    # SARIMA Forecast
    st.subheader("🔮 SARIMA Forecast")
    try:
        sarima_model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_fit = sarima_model.fit(disp=False)
        sarima_forecast = sarima_fit.forecast(steps=forecast_days)
        st.line_chart(pd.concat([df['Close'], sarima_forecast.rename("Forecast")]))
    except Exception as e:
        st.error(f"SARIMA Error: {e}")

    # Prophet Forecast
    st.subheader("🔮 Prophet Forecast")
    try:
        prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        fig = model.plot(forecast)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Prophet Error: {e}")
else:
    st.info("Please upload a stock CSV file to begin.")
