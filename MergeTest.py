from datetime import datetime
import pandas as pd
from tweety.bot import Twitter
from typing import List
from tweety.types import Tweet
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import pytz
from sklearn.ensemble import RandomForestRegressor



#database connection
conn = sqlite3.connect('test2.db')
cursor = conn.cursor()

#get tweets from db
cursor.execute("SELECT * FROM tweets")
columns = [column[0] for column in cursor.description]
rows = cursor.fetchall()
tweetsFromDb = pd.DataFrame(rows, columns=columns)

#get news from db
cursor.execute("PRAGMA table_info(news)")
columnsNews = [column[1] for column in cursor.fetchall()]

cursor.execute("SELECT * FROM news")
rows = cursor.fetchall()
newsFromDb = pd.DataFrame(rows, columns=columnsNews)

#get prices from db

    # Fetch the column names using PRAGMA statement
    cursor.execute("PRAGMA table_info(price_predictions)")
    columns = [column[1] for column in cursor.fetchall()]

    # Execute an SQL query to fetch the data from the table
    cursor.execute("SELECT * FROM price_predictions")
    rows = cursor.fetchall()

    #combine column Names and rows
    dfPrices = pd.DataFrame(rows, columns=columns)


#set datetime for all dataframes
dfPrices['date_time'] = pd.to_datetime(dfPrices['date_time'], utc=True)
dfPrices['date_time'] = dfPrices['date_time'].dt.floor('T')

tweetsFromDb['created_at'] = pd.to_datetime(tweetsFromDb['created_at'])
tweetsFromDb['created_at'] = tweetsFromDb['created_at'].dt.floor('T')

newsFromDb['pub_date'] =pd.to_datetime(newsFromDb['pub_date'])
newsFromDb['pub_date'] = newsFromDb['pub_date'].dt.floor('T')

#convert Time zones
berlin_tz = pytz.timezone('Europe/Berlin')
newsFromDb['pub_date'] = newsFromDb['pub_date'].dt.tz_convert(berlin_tz)
tweetsFromDb['created_at'] = tweetsFromDb['created_at'].dt.tz_convert(berlin_tz)
dfPrices['date_time'] = dfPrices['date_time'].dt.tz_convert(berlin_tz)


    #TEST DATA NOT IN DATABASE YET - ETH-USD prices
        dfPrices = pd.read_csv('ETHUSD_1.csv')
        dfPrices.rename(columns={"1438956180": "Date", "3.0": "Open","3.0.1": "High", "3.0.2": "Low", "3.0.3": "Close", "81.85727776": "Volume", "2": "Trades"}, inplace=True)
        dfPrices['Date'] = pd.to_datetime(dfPrices['Date'], unit='s')
        dfPrices['Date'] = pd.to_datetime(dfPrices['Date'], utc=True)
        dfPrices['Date'] = dfPrices['Date'].dt.floor('T')
        dfPrices['Date'] = dfPrices['Date'].dt.tz_convert(berlin_tz)
        #merge all 3 dataframes
        merged_df1 = pd.merge(tweetsFromDb, dfPrices, left_on='created_at', right_on='Date')
        merged_df2 = pd.merge(merged_df1, newsFromDb, left_on='created_at', right_on='pub_date')

        merged_df1 = merged_df1.drop(columns=['id', 'text_tweet'])

#merge all 3 dataframes
merged_df1 = pd.merge(tweetsFromDb, dfPrices, left_on='created_at', right_on='date_time')
merged_df2 = pd.merge(merged_df1, newsFromDb, left_on='created_at', right_on='pub_date')


#Vectorize tweets
count_vect = CountVectorizer(min_df=1)
X = count_vect.fit_transform(merged_df1["text_tweet"])
tweetTxt = merged_df1["text_tweet"]
dtm = pd.DataFrame(X.toarray())
dtm.columns = count_vect.get_feature_names_out()
data_dtm = pd.concat([merged_df1.reset_index(drop=True), dtm], axis=1)

#
train_size = int(len(data_dtm) * 0.8)  # 80% of the data for training
train_data = data_dtm[:train_size]
test_data = data_dtm[train_size:]


# Scale the 'Close' prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))

# Create the input sequences and corresponding labels
prediction_minutes = 60
x_train, y_train = [], []

for i in range(prediction_minutes, len(scaled_train_data)):
    x_train.append(scaled_train_data[i - prediction_minutes:i, 0])
    y_train.append(scaled_train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#train model
model_rf = RandomForestRegressor(n_estimators=100, random_state=0)
model_rf.fit(x_train, y_train)



#PREPARE TESTING DATA
# Scale the 'Close' prices for testing data
scaled_test_data = scaler.transform(test_data['Close'].values.reshape(-1, 1))

# Create the input sequences for testing
x_test, y_test = [], []
for i in range(prediction_minutes, len(scaled_test_data)):
    x_test.append(scaled_test_data[i - prediction_minutes:i, 0])
    y_test.append(scaled_test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

#Predict
predictions = model_rf.predict(x_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

profit_factor = np.sum(y_test[np.where(predictions > y_test)]) / np.abs(np.sum(y_test[np.where(predictions < y_test)]))
