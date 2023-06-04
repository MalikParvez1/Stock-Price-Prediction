import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from bs4 import BeautifulSoup
from googlesearch import search
import yfinance as yf
from database import create_connection, create_tables, insert_price_prediction, insert_news

# Define Google News scraping function
def scrape_google_news(query):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    articles = soup.findAll('item')
    news = []

    for article in articles:
        title = article.title.text
        link = article.link.text
        pub_date = article.pubDate.text
        news.append({
            'title': title,
            'link': link,
            'pub_date': pub_date
        })

    return news


crypto_currency = 'BTC'
against_currency = 'USD'

# Specify the date range for data retrieval
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

# Fetch historical price data
data = yf.download(tickers=f'{crypto_currency}-{against_currency}', period='7d', interval='1m')

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

# Create Neural Network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Fetch news from Google News
query = f'{crypto_currency} news'  # Query to search for news
news_headlines = scrape_google_news(query)

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

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, color='black', label='Actual Prices')
plt.plot(prediction_prices, color='green', label='Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Verbindung zur Datenbank herstellen
conn = create_connection("database.db")

# Tabellen erstellen
create_tables(conn)

# Speichere die Preisvorhersagen in der Datenbank
for i in range(len(actual_prices)):
    actual_price = actual_prices[i]
    prediction_price = prediction_prices[i]

    # Rufe die Funktion insert_price_prediction auf, um die Daten in der Datenbank zu speichern
    insert_price_prediction(conn, actual_price, prediction_price)

# Speichere die Google News in der Datenbank
for headline in news_headlines:
    title = headline['title']
    link = headline['link']
    pub_date = headline['pub_date']

    # Rufe die Funktion insert_news auf, um die Daten in der Datenbank zu speichern
    insert_news(conn, title, link, pub_date)

# Verbindung zur Datenbank schließen
conn.close()
