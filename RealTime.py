
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.title('Stock Trend Prediction with Hold/Sell Suggestion and Quantity')

# User input for stock ticker and available capital
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
capital = st.number_input('Enter Available Capital (USD)', min_value=100.0, value=1000.0)
share_price_input = st.number_input('Enter Number of Shares Held', min_value=0, value=0)

try:
    # Fetch the latest stock data
    df = yf.download(user_input, period='1d', interval='1m')

    # Check if data is empty
    if df.empty:
        st.subheader(f"No data available for ticker '{user_input}'. Please check the ticker and try again.")
    else:
        # Displaying data
        st.subheader('Latest Stock Data')
        st.dataframe(df.tail())  # Show latest fetched data

        # Visualize closing price over time
        st.subheader('Closing Price vs Time chart')
        fig = plt.figure(figsize=(12, 7))
        plt.plot(df['Close'], label='Closing Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)

        # Additional Feature Calculation
        df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()  # 20-day Exponential Moving Average

        # RSI Calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD Calculation (12-day EMA - 26-day EMA)
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']

        # Bollinger Bands Calculation
        df['20_SMA'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper Band'] = df['20_SMA'] + (df['STD'] * 2)
        df['Lower Band'] = df['20_SMA'] - (df['STD'] * 2)

        # Dropping NaN values
        df.dropna(inplace=True)

        # Display the new features
        st.subheader('Technical Indicators')
        st.dataframe(df[['SMA_20', 'EMA_20', 'RSI', 'MACD', 'Upper Band', 'Lower Band']].tail())

        # Prepare data for prediction (Only 'Close' price is used here)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        # Reshape for model input (last 100 time steps)
        x_input = scaled_data[-100:].reshape(1, -1, 1)  # Only 1 feature (Close price)

        # Load pre-trained model
        model = load_model('StockPrediction.keras')

        # Make prediction
        y_predicted = model.predict(x_input)

        # Inverse transform to get actual predicted value
        scale_factor = 1 / scaler.scale_[0]
        predicted_price = y_predicted * scale_factor

        st.subheader('Predicted Stock Price')
        st.write(f"The predicted closing price is: ${predicted_price[0][0]:.2f}")

        # Get the current price
        current_price = df['Close'].iloc[-1]

        # Define rules for suggesting Hold or Sell
        suggestion = ""
        if df['RSI'].iloc[-1] > 70:  # Overbought
            suggestion = "Sell (RSI suggests overbought conditions)"
        elif df['RSI'].iloc[-1] < 30:  # Oversold
            suggestion = "Hold (RSI suggests oversold conditions)"
        elif current_price > df['SMA_20'].iloc[-1]:  # Price above SMA
            suggestion = "Hold (Price above 20-day SMA)"
        elif current_price < df['SMA_20'].iloc[-1]:  # Price below SMA
            suggestion = "Sell (Price below 20-day SMA)"
        elif predicted_price > current_price:  # Predicted price higher than current
            suggestion = "Hold (Predicted price higher than current)"
        elif predicted_price < current_price:  # Predicted price lower than current
            suggestion = "Sell (Predicted price lower than current)"

        st.subheader('Suggested Action Based on Indicators')
        st.write(suggestion)

        # Suggest Quantity of Shares
        shares_to_trade = 0
        if "Hold" in suggestion:
            # Calculate max shares user can buy with 10% of their capital
            shares_to_trade = (0.1 * capital) // current_price
            st.write(f"Suggested to hold. You can buy up to {shares_to_trade:.0f} shares with 10% of your capital (${capital}).")
        elif "Sell" in suggestion:
            # Suggest selling a portion of the current shares
            shares_to_trade = share_price_input * 0.5  # Sell 50% of current holdings
            st.write(f"Suggested to sell. Consider selling {shares_to_trade:.0f} shares (50% of your current holdings).")

except Exception as e:
    st.error(f"Error fetching data for ticker '{user_input}'. Please check the ticker and try again. Details: {e}")

