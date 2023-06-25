from tweety.bot import Twitter
from typing import List
from tweety.types import Tweet
import pandas as pd
from datetime import datetime
import datetime as dt
import yfinance as yf
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

from database import create_connection, create_tables, insert_tweet, check_tweet_exists, check_price_prediction_exists, \
    insert_price_prediction

app = Twitter()

def create_dataframe_from_tweets(tweets: List[Tweet]) -> pd.DataFrame:
    data = []
    for tweet in tweets:
        if len(tweet.text) == 0:
            continue
        data.append({
            "id": tweet.id,
            "text_tweet": tweet.text,
            "author": tweet.author.username,
            "created_at": tweet.date,
            "views": tweet.views,
            "length": len(tweet.text)
        })
    df = pd.DataFrame(data, columns=["id", "text_tweet", "author", "date", "views", "created_at"])
    df.set_index("id", inplace=True)
    if df.empty:
        return df
    df = df[df.created_at.dt.date > datetime.now().date() - pd.to_timedelta("365day")]
    return df

def fetch_tweets(usernames: List[str]):
    # Verbindung zur Datenbank herstellen
    conn = create_connection("test2.db")

    # Tabellen erstellen
    create_tables(conn)

    # Tweets von den angegebenen Benutzern abrufen
    all_tweets = []
    counter = 0;
    for username in usernames:
        print(username)
        tweets = app.get_tweets(username=username, pages=100)
        for tweet in tweets:
            counter= counter+1
            print(counter)
            tweet_text = tweet.text.lower()
            all_tweets.append(tweet)

    # Tweets in DataFrame umwandeln
    tweet_df = create_dataframe_from_tweets(all_tweets)
    print(tweet_df)

    # Tweets in die Datenbank speichern
    for tweet in all_tweets:
        if not check_tweet_exists(conn, tweet.text, tweet.author.username, tweet.date, tweet.views, len(tweet.text)):
            insert_tweet(conn, tweet.text, tweet.author.username, tweet.date, tweet.views, len(tweet.text))
            print("TWEET IS NEW!")

    # Datenbankverbindung schließen
    conn.close()



# Liste der Benutzernamen, von denen Tweets abgerufen werden sollen
#usernames = ["VitalikButerin","elonmusk", "ErikVoorhees","Sassal0x", "rogerkver", "APompliano", "cz_binance", "scottmelker", "TheCryptoLark", "TimDraper", "SatoshiLite", "balajis", "brian_armstrong", "WuBlockchain", "woonomic", "CryptoWendyO", "MMCrypto", "100trillionUSD", "girlgone_crypto", "CryptoCred" ]
usernames = ["VitalikButerin","elonmusk", "ErikVoorhees"]
fetch_tweets(usernames)

usernames = ["Sassal0x", "rogerkver", "APompliano"]
fetch_tweets(usernames)

usernames = ["scottmelker", "TheCryptoLark", "TimDraper", "SatoshiLite", "balajis", "brian_armstrong", "WuBlockchain", "woonomic", "CryptoWendyO", "MMCrypto", "100trillionUSD", "girlgone_crypto", "CryptoCred"]
fetch_tweets(usernames)

usernames = ["DefiIgnas","Excellion", "DylanLeClair_", "CryptoWendyO", "CryptoHayes"]
fetch_tweets(usernames)

usernames= ["novogratz","glassnode","maxkeiser", "PeterMcCormack", "danheld", "WClementeIII", "elliotrades"]
fetch_tweets(usernames)

usernames= ["RaoulGMI", "TheMoonCarl", "saylor"]
fetch_tweets(usernames)

time.sleep(5)  # Pause von 1 Minute



#HISTORICAL PRICES
def insert_historical_Prices_into_db():
    conn = create_connection("test2.db")
    dfPrices = pd.read_csv('ETHUSD_1.csv')
    dfPrices.rename(columns={"1438956180": "Date", "3.0": "Open", "3.0.1": "High", "3.0.2": "Low", "3.0.3": "Close",
                             "81.85727776": "Volume", "2": "Trades"}, inplace=True)
    dfPrices['Date'] = pd.to_datetime(dfPrices['Date'], unit='s')
    with open('ETHUSD_1.csv', 'r') as file:
        for _, row in dfPrices.iterrows():
            date_time = str(row['Date'])
            actual_price = row['Close']
            predicted_price = None

            if not check_price_prediction_exists(conn, actual_price, predicted_price):
                insert_price_prediction(conn, date_time, actual_price, predicted_price)
                print("Historical Data added-")
            else:
                print("Price prediction already exists")

insert_historical_Prices_into_db()





