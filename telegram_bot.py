import os
import pandas as pd
import numpy as np
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load or fetch data functions
def fetch_detailed_historical_data(currency_pair="ME_USDT",
                                   interval="1h",
                                   limit=500):
    api_url = "https://api.gateio.ws/api/v4/spot/candlesticks"
    params = {
        "currency_pair": currency_pair,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        candlestick_data = response.json()
        df = pd.DataFrame(candlestick_data)
        df.columns = [
            "time", "volume", "close", "high", "low", "open", "quote_volume",
            "change_rate"
        ]
        df['time'] = pd.to_numeric(df['time'])
        df['date'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('date').drop(columns=['time', 'change_rate'])
        for col in ["close", "open", "high", "low", "volume", "quote_volume"]:
            df[col] = pd.to_numeric(df[col])
        return df
    else:
        raise Exception(
            f"Failed to fetch data. HTTP Status Code: {response.status_code}")


def update_dataset(new_data, file_path="historical_data.csv"):
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path, parse_dates=['date'])
        combined_data = pd.concat(
            [existing_data,
             new_data]).drop_duplicates(subset=['date']).sort_values('date')
    else:
        combined_data = new_data
    combined_data.to_csv(file_path, index=False)
    return combined_data


# Preprocess functions
def preprocess_data(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset['scaled_close'] = scaler.fit_transform(dataset[['close']])
    return dataset, scaler


def create_sequences(data, seq_length=7):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# Model building and prediction
def build_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    return model


def predict_next_days(model, recent_data, scaler, days=7):
    predictions = []
    data = recent_data.copy()
    for _ in range(days):
        prediction = model.predict(data.reshape(1, -1, 1))
        predictions.append(prediction[0, 0])
        data = np.append(data[1:], prediction)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


# Telegram bot handlers
async def start(update: Update, context: CallbackContext.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to ME-USDT Bot! Use /predict to get the 7-day prediction of ME-USDT prices."
    )


async def predict(update: Update, context: CallbackContext.DEFAULT_TYPE):
    await update.message.reply_text(
        "Fetching and processing data. This may take a moment...")

    try:
        # Fetch and update dataset
        new_data = fetch_detailed_historical_data()
        dataset = update_dataset(new_data)
        dataset, scaler = preprocess_data(dataset)

        # Prepare sequences
        seq_length = 7
        X, y = create_sequences(dataset['scaled_close'].values, seq_length)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Load or train the model
        model_path = "advanced_lstm_model.keras"
        if os.path.exists(model_path):
            model = load_model(model_path, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='mean_squared_error')
        else:
            model = build_model(input_shape=(seq_length, 1))
            model.fit(X, y, epochs=50, batch_size=16, verbose=1)
            model.save(model_path)

        # Make predictions
        recent_sequence = X[-1]
        future_predictions = predict_next_days(model,
                                               recent_sequence,
                                               scaler,
                                               days=7)

        # Prepare response
        last_date = dataset['date'].iloc[-1]
        next_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
        prediction_message = "Next 7 Days Predictions:\n"
        for date, price in zip(next_dates, future_predictions.flatten()):
            prediction_message += f"{date.strftime('%Y-%m-%d')}: {price:.2f}\n"

        await update.message.reply_text(prediction_message)
    except Exception as e:
        await update.message.reply_text(f"An error occurred: {e}")


# Main bot setup
if __name__ == "__main__":
    TOKEN = "8041657061:AAE-G5B1k3yubYvJ4rqDvzp4r-91Vwcl6-I"
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))

    print("Bot is running...")
    app.run_polling()
