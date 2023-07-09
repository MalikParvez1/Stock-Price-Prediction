from tweety.bot import Twitter
from typing import List
from tweety.types import Tweet
import pandas as pd
from datetime import datetime
import datetime as dt
import yfinance as yf
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import nltk
nltk.download('vader_lexicon')


from database import create_connection, create_tables, insert_tweet, check_tweet_exists

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
    counter = 0
    for username in usernames:
        tweets = app.get_tweets(username=username, pages=1)
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

    #Split and Count Tweets words
    count_vect = CountVectorizer(min_df=1)
    X = count_vect.fit_transform(tweet_df["text_tweet"])
    tweetTxt = tweet_df["text_tweet"]
    dtm = pd.DataFrame(X.toarray())
    dtm.columns = count_vect.get_feature_names_out()
    data_dtm = pd.concat([tweet_df.reset_index(drop=True), dtm], axis=1)

    #leave out the time of the tweets DATE column
    data_dtm['created_at'] = pd.to_datetime(data_dtm['created_at'].apply(datetime.date))
    data_dtm = data_dtm.drop('date', axis=1)


    # sentiment
    sid = SentimentIntensityAnalyzer()
    tweet_df['sentiment'] = tweet_df['text_tweet'].apply(lambda x: sid.polarity_scores(x))
    print(tweet_df[['text_tweet', 'sentiment']])

# Liste der Benutzernamen, von denen Tweets abgerufen werden sollen
#usernames = ["VitalikButerin","elonmusk", "ErikVoorhees","Sassal0x", "rogerkver", "APompliano", "cz_binance", "scottmelker", "TheCryptoLark", "TimDraper", "SatoshiLite", "balajis", "brian_armstrong", "WuBlockchain", "woonomic", "CryptoWendyO", "MMCrypto", "100trillionUSD", "girlgone_crypto", "CryptoCred" ]
usernames = ["VitalikButerin", "Sassal0x", "rogerkver", "APompliano", "scottmelker", "TheCryptoLark", "TimDraper", "SatoshiLite", "balajis", "brian_armstrong", "WuBlockchain", "woonomic", "CryptoWendyO", "MMCrypto", "100trillionUSD", "girlgone_crypto", "CryptoCred",
             "DefiIgnas","Excellion", "DylanLeClair_", "CryptoWendyO", "CryptoHayes","novogratz","glassnode","maxkeiser", "PeterMcCormack", "danheld", "WClementeIII", "elliotrades","RaoulGMI", "TheMoonCarl", "saylor"]

# Endlosschleife, um Tweets alle 1 Minute abzurufen
while True:
    fetch_tweets(usernames)
    time.sleep(60)  # Pause von 1 Minute
