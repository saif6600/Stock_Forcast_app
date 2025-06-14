import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Streamlit Page Config
st.set_page_config(page_title="📊 Stock Forecast App", layout="wide")
st.title("📈 Stock Price Forecasting")
st.markdown("Upload a stock CSV file to generate forecasts using ARIMA, SARIMA, Prophet, and LSTM models.")

# File Upload
file = st.file_uploader("📤 Upload CSV File", type=["csv"])

# Forecast Slider
forecast_days = st.slider("Select Forecast Days", min_value=7, max_value=90, value=30)

# Main logic
if file:
    # Load and clean data
    df = pd.read_csv(file, skiprows=2)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(inplace=True)
    df.set_index('Date', inplace=True)
    st.line_chart(df['Close'])

    #### ARIMA ####
    st.subheader("🔮 ARIMA Forecast")
    try:
        arima_model = ARIMA(df['Close'], order=(5, 1, 0))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=forecast_days)
        arima_result = pd.concat([df['Close'], arima_forecast.rename("ARIMA Forecast")])
        st.line_chart(arima_result)

        arima_df = pd.DataFrame({
            'Date': arima_forecast.index,
            'Predicted_Close': arima_forecast.values
        })
        arima_csv = arima_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download ARIMA Forecast", arima_csv, "arima_forecast.csv", "text/csv")
    except Exception as e:
        st.error(f"ARIMA Error: {e}")

    #### SARIMA ####
    st.subheader("🔮 SARIMA Forecast")
    try:
        sarima_model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_fit = sarima_model.fit(disp=False)
        sarima_forecast = sarima_fit.forecast(steps=forecast_days)
        sarima_result = pd.concat([df['Close'], sarima_forecast.rename("SARIMA Forecast")])
        st.line_chart(sarima_result)

        sarima_df = pd.DataFrame({
            'Date': sarima_forecast.index,
            'Predicted_Close': sarima_forecast.values
        })
        sarima_csv = sarima_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download SARIMA Forecast", sarima_csv, "sarima_forecast.csv", "text/csv")
    except Exception as e:
        st.error(f"SARIMA Error: {e}")

    #### Prophet ####
    st.subheader("🔮 Prophet Forecast")
    try:
        prophet_data = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        prophet_model = Prophet()
        prophet_model.fit(prophet_data)
        future = prophet_model.make_future_dataframe(periods=forecast_days)
        forecast = prophet_model.predict(future)

        fig1 = prophet_model.plot(forecast)
        st.pyplot(fig1)

        prophet_forecast = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted_Close'})
        prophet_csv = prophet_forecast[-forecast_days:].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prophet Forecast", prophet_csv, "prophet_forecast.csv", "text/csv")
    except Exception as e:
        st.error(f"Prophet Error: {e}")

    #### LSTM ####
    st.subheader("🤖 LSTM Forecast (Deep Learning)")
    try:
        close_data = df[['Close']]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)

        def create_dataset(dataset, time_step=60):
            X, y = [], []
            for i in range(len(dataset) - time_step - 1):
                X.append(dataset[i:(i + time_step), 0])
                y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 60
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

        # Forecasting future values
        temp_input = list(scaled_data[-time_step:].flatten())
        lstm_output = []

        for _ in range(forecast_days):
            input_seq = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
            pred = model.predict(input_seq, verbose=0)[0][0]
            temp_input.append(pred)
            lstm_output.append(pred)

        lstm_forecast = scaler.inverse_transform(np.array(lstm_output).reshape(-1, 1))
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        lstm_series = pd.Series(lstm_forecast.flatten(), index=future_dates)

        # Plot
        fig2, ax2 = plt.subplots()
        ax2.plot(df['Close'], label="Actual")
        ax2.plot(lstm_series, label="LSTM Forecast")
        ax2.legend()
        st.pyplot(fig2)

        lstm_df = pd.DataFrame({
            'Date': lstm_series.index,
            'Predicted_Close': lstm_series.values
        })
        lstm_csv = lstm_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download LSTM Forecast", lstm_csv, "lstm_forecast.csv", "text/csv")
    except Exception as e:
        st.error(f"LSTM Error: {e}")
else:
    st.info("Please upload a CSV file to begin.")
