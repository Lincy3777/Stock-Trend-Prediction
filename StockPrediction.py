import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2021-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

#Describing Data
st.subheader('Data from 2010 - 2021')
st.dataframe(df.describe())

#Visualization
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,7))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('closing price')
plt.legend()
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])#from start till 70% of the data
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])#from end of 70% of the data till last

#normalizing the df
scaler = MinMaxScaler(feature_range=(0,1))#Scale between 0 and 1
data_training_array = scaler.fit_transform(data_training)#scaler.fit_transform will give a array

#Load model 
model =load_model('StockPrediction.keras')

#tesing part
past_100_days = data_training.tail(100)#I need this, inorder to predict the first value of testing data 
#final df
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)#ignore_index is set true because it helps to reset the df and maintains uniqueness 
input_data = scaler.fit_transform(final_df)
x_test= []
y_test = []

for i in range(100, input_data.shape[0]):#till 1006
  x_test.append(input_data[i-100:i])#100 days before the current day(i)
  y_test.append(input_data[i,0])#0 is the closing price column, appending the value of closing price in the y_test 

x_test, y_test = np.array(x_test), np.array(y_test)

#Making prediction
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('prediction vs original')
fig2 = plt.figure(figsize=(12,7))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Predicted price')
plt.ylabel('Original price')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 =df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,7))
plt.plot(ma100)
plt.plot(df.Close)
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA and 100MA')
ma100 =df.Close.rolling(100).mean()
ma200 =df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,7))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend()
st.pyplot(fig)

# def get_suggestion(ma100, ma200):
#     if ma100.iloc[-1] > ma200.iloc[-1]:#acessing the last rows of both columns using integer location
#         return "Hold"
#     else:
#         return "Sell"
 
ma100_latest = ma100.iloc[-1]
ma200_latest = ma200.iloc[-1]

# Get the last predicted price
last_predicted_price = y_predicted[-1]

def get_suggestion(ma100_latest, ma200_latest, last_predicted_price):
    if last_predicted_price > ma100_latest and ma100_latest > ma200_latest:
        return "Buy"
    elif last_predicted_price < ma100_latest and ma100_latest < ma200_latest:
        return "Sell"
    else:
        return "Hold"

# Add the suggestion
if len(df) >= 200:
    suggestion = get_suggestion(ma100_latest, ma200_latest, last_predicted_price)
    st.subheader(f"Suggestion: {suggestion}")
else:
    st.subheader("Not enough data for suggestion (Need at least 200 days)")

