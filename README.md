# 📈 Stock Market Prediction with LSTM + Reinforcement Learning

This project implements a **hybrid intelligent trading platform** that combines **LSTM-based stock price prediction** with **Deep Q-Learning (DQN)** for dynamic trading decisions. It provides a user-friendly interface via **Streamlit**, allowing users to test predictions and strategies for any stock ticker supported by Yahoo Finance.

---

## 🚀 Features

- 🧠 **LSTM** model trained on historical price data to predict trends.
- 🤖 **Reinforcement Learning** (DQN) to make Buy/Sell/Hold decisions.
- 🔀 **Hybrid Decision System**: Combines LSTM predictions and RL policies.
- 📊 **Streamlit Web App** for easy interaction and visualization.
- 🔄 Works for **any stock ticker**: AAPL, NVDA, TSLA, GOOGL, etc.
- 📁 Automatic local caching of CSV files via `yfinance`.

---

## 📁 Project Structure
├── Streamlit_code.py         # Streamlit application for stock trading interface \
├── AAPL.csv                  # Downloaded stock data (generated when a stock ticker is entered) \
├── stock_trading_env.py      # Custom OpenAI Gym trading environment \
├── LSTM.h5                   # Trained LSTM model file \
├── requirements.txt          # Python dependencies \
└── README.md                 # Project documentation 

---

## Running the web app

streamlit run streamlit_code.py

---

## Example usage

Once the app is running:
	1.	Enter a stock ticker (e.g., AAPL, NVDA, TSLA)
	2.	The app:
	•	Downloads and cleans the stock data
	•	Scales it and feeds it to the LSTM model
	•	Predicts future price trends
	•	Simulates trading using the RL agent
	•	Displays total account value over time

 
