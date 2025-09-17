# LSTM for Stock and Options Price Prediction
# Disclaimer: This script is for educational purposes only and is not financial advice.
# Financial markets are highly volatile and unpredictable. Past performance is not indicative of future results.

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import mean_squared_error

# --- Part 1: Stock Price Prediction ---

def fetch_stock_data(ticker, start_date, end_date):
    """Fetches stock data from Yahoo Finance."""
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        raise ValueError(f"No data fetched for ticker {ticker}. Check the ticker symbol and date range.")
    print("Data fetched successfully.")
    return stock_data

def preprocess_stock_data(data):
    """Preprocesses the stock data for the LSTM model."""
    print("Preprocessing stock data...")
    # Using 'Close' price for prediction
    close_prices = data['Close'].values.reshape(-1, 1)

    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    # Create sequences of data
    X, y = [], []
    sequence_length = 60  # Use 60 days of historical data to predict the next day

    for i in range(sequence_length, len(scaled_prices)):
        X.append(scaled_prices[i - sequence_length:i, 0])
        y.append(scaled_prices[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshape data for LSTM [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print("Preprocessing complete.")
    return X, y, scaler

def build_lstm_model(input_shape):
    """Builds the LSTM model architecture."""
    print("Building LSTM model...")
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1)) # Prediction of the next closing value

    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Model built successfully.")
    model.summary()
    return model

def plot_stock_predictions(actual_prices, predicted_prices, ticker):
    """Plots the actual vs. predicted stock prices."""
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, color='cyan', label=f'Actual {ticker} Price')
    plt.plot(predicted_prices, color='magenta', linestyle='--', label=f'Predicted {ticker} Price')
    plt.title(f'{ticker} Stock Price Prediction', fontsize=16)
    plt.xlabel('Time (Days)', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def run_stock_prediction():
    """Main function to run the stock price prediction."""
    # Parameters
    TICKER = 'AAPL'
    START_DATE = '2015-01-01'
    END_DATE = date.today().strftime('%Y-%m-%d') # Get today's date as end date
    TRAIN_SPLIT_RATIO = 0.8

    # 1. Fetch Data
    try:
        stock_data = fetch_stock_data(TICKER, START_DATE, END_DATE)
    except ValueError as e:
        print(e)
        return

    # 2. Preprocess Data
    X, y, scaler = preprocess_stock_data(stock_data)

    # 3. Split data into training and testing sets
    training_size = int(len(X) * TRAIN_SPLIT_RATIO)
    X_train, X_test = X[:training_size], X[training_size:]
    y_train, y_test = y[:training_size], y[training_size:]

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # 4. Build and Train Model
    model = build_lstm_model((X_train.shape[1], 1))

    print("Training model...")
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)
    print("Model training complete.")

    # 5. Make Predictions
    print("Making predictions on the test set...")
    predicted_scaled_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_scaled_prices)

    # 6. Evaluate and Plot
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    plot_stock_predictions(actual_prices, predicted_prices, TICKER)

    # 7. Evaluate Model Performance
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    print(f"\nModel Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# --- Part 2: Options Price Prediction (Conceptual) ---

def conceptual_options_prediction():
    """
    This function outlines the conceptual steps for options price prediction.
    Predicting options prices is significantly more complex than stocks due to multiple,
    non-linear factors (the "Greeks", implied volatility, etc.). A robust model
    would require a much larger and more complex dataset and architecture.
    """
    print("\n--- Conceptual Outline for Options Price Prediction ---")
    print("1. Data Collection: You would need historical data for:")
    print("   - Underlying stock price (e.g., AAPL Close)")
    print("   - Option strike price")
    print("   - Time to expiration (in years/days)")
    print("   - Implied Volatility (IV)")
    print("   - Risk-free interest rate (e.g., US Treasury yield)")
    print("   - The actual option's historical price (the target variable)")

    print("\n2. Feature Engineering & Preprocessing:")
    print("   - All features must be numeric and scaled (e.g., using MinMaxScaler).")
    print("   - Create sequences just like in the stock prediction example, but now each timestep will have multiple features.")
    print("   - Example Input Shape: (samples, timesteps, 5) for the 5 features above.")

    print("\n3. Model Adaptation:")
    print("   - The LSTM model architecture would be similar, but the input_shape in the first layer must be adjusted to match the new number of features.")
    print("   - `model.add(LSTM(..., input_shape=(sequence_length, num_features)))`")

    print("\n4. Challenges & Advanced Approaches:")
    print("   - High non-linearity and sensitivity to market news.")
    print("   - Data Sparsity: Options contracts are not traded as frequently as stocks.")
    print("   - A more common and potentially more effective approach is to predict Implied Volatility (IV) first, and then use it in a pricing model like Black-Scholes to derive the option price.")
    print("-----------------------------------------------------\n")


if __name__ == '__main__':
    # Run the stock prediction workflow
    run_stock_prediction()

    # Display the conceptual outline for options prediction
    conceptual_options_prediction()
