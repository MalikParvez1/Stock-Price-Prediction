import requests
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Konstanten
invest_amount = 10000  # Investitionsbetrag in Euro
api_url = "https://api.coindesk.com/v1/bpi/historical/close.json?start=2021-07-17&end=2023-06-29"  # API-URL für historische Bitcoin-Preise

# Funktion zum Abrufen der historischen Bitcoin-Preise
def get_historical_prices():
    response = requests.get(api_url)
    data = response.json()
    return data["bpi"]

# Funktion zum Erstellen von Sequenzen für das LSTM-Modell
def create_sequences(data, target):
    X = []
    y = []
    for i in range(len(data)-1):
        X.append(data[i])
        y.append(target[i+1])
    return np.array(X), np.array(y)

# Funktion zum Investieren in Bitcoins
def invest_in_bitcoins():
    prices = get_historical_prices()
    dates = list(prices.keys())
    prices_list = list(prices.values())

    # LSTM-Modell zum Bitcoin-Preisvorhersage erstellen
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(np.array(prices_list).reshape(-1, 1))

    train_size = int(len(prices_scaled) * 0.8)
    train_data = prices_scaled[:train_size]
    test_data = prices_scaled[train_size:]

    X_train, y_train = create_sequences(train_data, train_data)
    X_test, y_test = create_sequences(test_data, test_data)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=10, batch_size=1)

    # Vorhersagen mit dem LSTM-Modell generieren
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Restlicher Code für die Investitionsstrategie
    bitcoins = 0
    total_invested = 0
    total_sold = 0
    remaining_bitcoins = 0

    for i in range(1, len(prices_list)):
        current_price = prices_list[i]
        previous_price = prices_list[i - 1]

        if current_price > previous_price:
            bitcoins_purchased = invest_amount / previous_price
            bitcoins += bitcoins_purchased
            total_invested += invest_amount
            print("{}: Gekauft - {} Bitcoins".format(dates[i], bitcoins_purchased))

        elif current_price < previous_price:
            bitcoins_sold = bitcoins - remaining_bitcoins
            total_sold += bitcoins_sold
            total_invested -= bitcoins_sold * current_price
            remaining_bitcoins = bitcoins
            print("{}: Verkauft - {} Bitcoins".format(dates[i], bitcoins_sold))

    final_balance = remaining_bitcoins * prices_list[-1] + total_invested
    total_profit_loss = final_balance - invest_amount
    remaining_balance = final_balance - total_sold * prices_list[-1]
    depot_balance = total_sold * prices_list[-1]

    print("Endgültiger Kontostand: {:.2f} Euro".format(final_balance))
    print("Gesamtgewinn/-verlust: {:.2f} Euro".format(total_profit_loss))
    print("Depotguthaben: {:.2f} Euro".format(depot_balance))

    # Scatterplot der Vorhersagen und tatsächlichen Werte anzeigen
    plt.scatter(range(len(train_data)), train_data, color='blue', label='Actual (Train)')
    plt.scatter(range(len(train_data), len(train_data) + len(test_data)), test_data, color='green', label='Actual (Test)')
    plt.scatter(range(1, len(train_predictions) + 1), train_predictions, color='red', label='Predicted (Train)')
    plt.scatter(range(len(train_predictions) + 1, len(train_predictions) + 1 + len(test_predictions)), test_predictions, color='magenta', label='Predicted (Test)')
    plt.xlabel('Time')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.show()

# Hauptprogramm
invest_in_bitcoins()
