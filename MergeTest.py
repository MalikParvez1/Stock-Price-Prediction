from datetime import datetime

import pandas as pd
from tweety.bot import Twitter
from typing import List
from tweety.types import Tweet


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

tweets = app.get_tweets(username="elonmusk", pages=100)
tweets_df = create_dataframe_from_tweets(tweets)
prices_df = pd.read_csv('ETHUSD_1.csv')
prices_df.rename(columns={"1438956180": "Date", "3.0": "Open","3.0.1": "High", "3.0.2": "Low", "3.0.3": "Close", "81.85727776": "Volume", "2": "Trades"}, inplace=True)
prices_df['Date'] = pd.to_datetime(prices_df['Date'], unit='s')

tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'])
tweets_df['created_at'] = tweets_df['created_at'].dt.floor('T')
prices_df['Date'] = prices_df['Date'].dt.floor('T')

prices_df['Date'] = pd.to_datetime(prices_df['Date']).dt.tz_localize(None)
tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at']).dt.tz_localize(None)



merged_df = pd.merge(tweets_df, prices_df, left_on='created_at', right_on='Date')
mer = pd.concat([tweets_df, prices_df], axis=1)

mm = tweets_df.set_index('created_at').join(prices_df.set_index('Date'))

