import streamlit as st
import pandas as pd
import os
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from Env import StockTradingEnvironment
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

s_model = tf.keras.models.load_model('LSTM.h5')


def get_stock_data(ticker):
    df1 = yf.download(ticker, start="2010-01-01", end="2024-01-01")
    df1.to_csv(f"{ticker}.csv")
    df = pd.read_csv(f"{ticker}.csv", skiprows=[1,2])
    df.rename(columns={"Price": "Date"})
    df.to_csv(f"{ticker}.csv",index=False)

def create_lstm_data(df, window_size=10):
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[i:i+window_size])
        y.append(df[i+window_size])
    return np.array(X), np.array(y)
def test_func(ticker, model):

    file_path = f"{ticker}.csv"
    if not os.path.exists(file_path):
        get_stock_data(ticker)

    df = pd.read_csv(file_path)

    prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()

    scaled_prices = scaler.fit_transform(prices)

    X,y= create_lstm_data(scaled_prices, window_size=10)

    predicted_prices = model.predict(X)
    y = y * (1/scaler.scale_)
    predicted_prices = predicted_prices * (1/scaler.scale_)

    plt.figure(figsize=(12,6))
    plt.plot(y,'b',label = 'Original price')
    plt.plot(predicted_prices,'r',label = 'Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Prices')
    plt.legend()
    st.pyplot(plt)
    
st.title("STOCK PREDICTION USING LSTM & DQN")    
ticker = st.text_input('Enter the ticker')
st.subheader("LSTM Prediction")
test_func(ticker,s_model)

stock_trading_environment = StockTradingEnvironment('./MSFT.csv', train=True ,number_of_days_to_consider=10)
stock_trading_environment.reset()
st.subheader("Prediction for the next 1000 days for a 1000$ investment")
model = DQN("MlpPolicy", stock_trading_environment, verbose=1)
model.learn(total_timesteps=1000)

st.pyplot(stock_trading_environment.render())