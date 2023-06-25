from datetime import datetime
import pandas as pd
from tweety.bot import Twitter
from typing import List
from tweety.types import Tweet
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import pytz
from sklearn.ensemble import RandomForestRegressor


def mergeDataframes():
    #database connection
    conn = sqlite3.connect('test2.db')
    cursor = conn.cursor()

    #get tweets from db
    cursor.execute("SELECT * FROM tweets")
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    tweetsFromDb = pd.DataFrame(rows, columns=columns)


    tweetsFromDb['created_at'] = pd.to_datetime(tweetsFromDb['created_at'])
    tweetsFromDb['created_at'] = tweetsFromDb['created_at'].dt.floor('T')


    #convert Time zones
    berlin_tz = pytz.timezone('Europe/Berlin')
    tweetsFromDb['created_at'] = tweetsFromDb['created_at'].dt.tz_convert(berlin_tz)


    #TEST DATA NOT IN DATABASE YET - ETH-USD prices
    dfPrices = pd.read_csv('ETHUSD_1.csv')
    dfPrices.rename(columns={"1438956180": "Date", "3.0": "Open","3.0.1": "High", "3.0.2": "Low", "3.0.3": "Close", "81.85727776": "Volume", "2": "Trades"}, inplace=True)
    dfPrices['Date'] = pd.to_datetime(dfPrices['Date'], unit='s')
    dfPrices['Date'] = pd.to_datetime(dfPrices['Date'], utc=True)
    dfPrices['Date'] = dfPrices['Date'].dt.floor('T')
    dfPrices['Date'] = dfPrices['Date'].dt.tz_convert(berlin_tz)
    #merge all 3 dataframes
    merged_df1 = pd.merge(tweetsFromDb, dfPrices, left_on='created_at', right_on='Date')
    #merged_df2 = pd.merge(merged_df1, newsFromDb, left_on='created_at', right_on='pub_date')



    #Vectorize tweets
    count_vect = CountVectorizer(min_df=1)
    X = count_vect.fit_transform(merged_df1["text_tweet"])
    tweetTxt = merged_df1["text_tweet"]
    dtm = pd.DataFrame(X.toarray())
    dtm.columns = count_vect.get_feature_names_out()
    data_dtm = pd.concat([merged_df1.reset_index(drop=True), dtm], axis=1)

    data_dtm = data_dtm.drop(columns=['id', 'text_tweet'])

    return data_dtm

mergeDataframes()

