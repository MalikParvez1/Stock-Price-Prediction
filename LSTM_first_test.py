import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import tensorflow
import yfinance as yf
from sklearn.metrics import r2_score
from database import create_connection, create_tables, insert_price_prediction, insert_news, check_price_prediction_exists, check_news_exists
import csv

# Assuming your CSV data is in the format: "YYYY-MM-DD,YYYY-MM-DD HH:MM:SS,actual_price,predicted_price"
def insert_newest_price_from_csv(conn, csv_file):
    try:
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)
            if len(rows) > 1:
                latest_row = rows[-1]
                date_time = latest_row[1]
                actual_price = float(latest_row[2])
                predicted_price = None
                if len(latest_row) >= 4:
                    predicted_price = float(latest_row[3])

                if not check_price_prediction_exists(conn, date_time, actual_price, predicted_price):
                    insert_price_prediction(conn, date_time, actual_price, predicted_price)

                print("Newest data inserted from CSV successfully")
            else:
                print("CSV file does not contain any new data")
    except FileNotFoundError:
        print("CSV file not found")
    except IndexError:
        print("Invalid CSV format")

# Usage example:
db_file = "database.db"
csv_file = 'Api/eth_price.csv'

# Connect to the database
conn = create_connection(db_file)

# Create tables if they don't exist
create_tables(conn)

# Insert the newest row from the CSV into the database
insert_newest_price_from_csv(conn, csv_file)

# Close the database connection
conn.close()



# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_minutes = 60

x_train, y_train = [], []

for x in range(prediction_minutes, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_minutes:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create LSTM Model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=1))

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(x_train, y_train, epochs=25, batch_size=32)

# Random Forest Regressor erstellen
model_rf = RandomForestRegressor(n_estimators=100, random_state=0)

# Anpassung der Form von x_train
x_train_rf = x_train.reshape(x_train.shape[0], x_train.shape[1])

# Modelltraining mit Random Forest Regressor
model_rf.fit(x_train_rf, y_train)

# Fetch news from Google News


# Perform further processing on the news headlines

# Fetch testing data
test_start = dt.datetime.now()
test_data = yf.download(tickers=f'{crypto_currency}-{against_currency}', start=test_start - dt.timedelta(days=2), end=test_start, interval='1m')
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_minutes:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []

for x in range(prediction_minutes, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_minutes:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Generate predictions using LSTM
prediction_prices_lstm = model_lstm.predict(x_test)
prediction_prices_lstm = scaler.inverse_transform(prediction_prices_lstm)

# Generate predictions using Random Forest Regressor
x_test_rf = x_test.reshape(x_test.shape[0], x_test.shape[1])
prediction_prices_rf = model_rf.predict(x_test_rf)
prediction_prices_rf = scaler.inverse_transform(prediction_prices_rf.reshape(-1, 1)).flatten()

plt.plot(actual_prices, color='black', label='Actual Prices')
plt.plot(prediction_prices_lstm, color='blue', label='LSTM Predicted Prices')
plt.plot(prediction_prices_rf, color='green', label='Random Forest Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

r2 = r2_score(actual_prices, prediction_prices_rf)
print('R2:',  r2)
# Simulating trades
virtual_account_balance = 10000  # Initial virtual account balance in USD
holding_coins = 0  # Number of coins held
buy_threshold = 0.0000  # Threshold for buying (2% increase in price)
sell_threshold = -0.0000  # Threshold for selling (2% decrease in price)
total_profit_loss = 0  # Track total profit/loss

for i in range(len(prediction_prices_rf)):
    predicted_price = prediction_prices_rf[i]
    actual_price = actual_prices[i]

    if predicted_price >= actual_price * (1 + buy_threshold):
        # Buy signal
        if holding_coins == 0:
            coins_to_buy = virtual_account_balance / actual_price
            holding_coins = coins_to_buy
            virtual_account_balance = 0
            print(f"Buy {coins_to_buy} coins at price {actual_price}")

    elif predicted_price <= actual_price * (1 + sell_threshold):
        # Sell signal
        if holding_coins > 0:
            coins_to_sell = holding_coins
            virtual_account_balance = coins_to_sell * actual_price
            holding_coins = 0
            print(f"Sell {coins_to_sell} coins at price {actual_price}")
            # Calculate profit/loss
            profit_loss = (virtual_account_balance - 10000) / 10000
            total_profit_loss += profit_loss

    # Print current balance and holdings
    print(f"Virtual Account Balance: {virtual_account_balance} USD")
    print(f"Holding Coins: {holding_coins} coins")

# Calculate percentage return on investment
percentage_return = (virtual_account_balance - 10000) / 10000 * 100

print("Total Profit/Loss:", total_profit_loss)
print("Percentage Return on Investment:", percentage_return, "%")

# Prediction for the next minute
next_minute_input = model_inputs[-prediction_minutes:].reshape(1, -1, 1)
next_minute_prediction_lstm = model_lstm.predict(next_minute_input)
next_minute_prediction_lstm = scaler.inverse_transform(next_minute_prediction_lstm)
next_minute_prediction_rf = model_rf.predict(next_minute_input.reshape(1, -1))
next_minute_prediction_rf = scaler.inverse_transform(next_minute_prediction_rf.reshape(-1, 1)).flatten()

print("_____________________________________________________")
print("_____________________________________________________")

print("Actual price:", actual_prices)
print("LSTM Predicted Price:", prediction_prices_lstm)
print("Random Forest Predicted Price:", prediction_prices_rf)

print("_____________________________________________________")
print("_____________________________________________________")

print("LSTM Predicted Price for the next minute:", next_minute_prediction_lstm[0][0])
print("Random Forest Predicted Price for the next minute:", next_minute_prediction_rf[0])

print("_____________________________________________________")
print("_____________________________________________________")

# Verbindung zur Datenbank herstellen
conn = create_connection("database.db")

# Tabellen erstellen
create_tables(conn)

# Speichere die Preisvorhersagen mit LSTM-Modell in der Datenbank
for i in range(len(actual_prices)):
    actual_price = actual_prices[i]
    prediction_price = prediction_prices_lstm[i]

    # Überprüfen, ob die Preisvorhersage bereits in der Datenbank vorhanden ist
    if not check_price_prediction_exists(conn, actual_price, prediction_price):
        # Rufe die Funktion insert_price_prediction auf, um die Daten in der Datenbank zu speichern
        insert_price_prediction(conn, actual_price, prediction_price)


# Verbindung zur Datenbank schließen
conn.close()