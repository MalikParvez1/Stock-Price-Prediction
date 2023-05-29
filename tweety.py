from tweety.bot import Twitter
from typing import List
from tweety.types import Tweet
import pandas as pd
from datetime import datetime
import datetime as dt
import yfinance as yf
from sklearn.feature_extraction.text import CountVectorizer

app = Twitter()

def create_dataframe_from_tweets(tweets: List[Tweet]) -> pd.DataFrame:
    data = []
    for tweet in tweets:
        if len(tweet.text) == 0:
            continue
        data.append({
            "id": tweet.id,
            "text": tweet.text,
            "author": tweet.author.username,
            "created_at": tweet.date,
            "views": tweet.views,
            "length": len(tweet.text)
        })
    df = pd.DataFrame(data, columns=["id", "text", "author", "date", "views", "created_at"])
    df.set_index("id", inplace=True)
    if df.empty:
        return df
    df = df[df.created_at.dt.date > datetime.now().date() - pd.to_timedelta("365day")]
    return df

all_tweets = app.get_tweets(username= "elonmusk", pages=100)
tweet_df = create_dataframe_from_tweets(all_tweets)
print(create_dataframe_from_tweets(all_tweets))

#Split and Count Tweets words
count_vect = CountVectorizer(min_df = 1)
X = count_vect.fit_transform(tweet_df["text"])
tweetTxt = tweet_df["text"]
dtm = pd.DataFrame(X.toarray())
dtm.columns = count_vect.get_feature_names_out()
data_dtm = pd.concat([tweet_df.reset_index(drop=True), dtm], axis=1)

#leave out the time of the tweets DATE column
data_dtm['created_at'] = pd.to_datetime(data_dtm['created_at'].apply(datetime.date))
data_dtm = data_dtm.drop('date', axis=1)



#CRYPTO PRICES

crypto_currency = 'ETH'
against_currency = 'USD'

eth_data = yf.download(tickers=f'{crypto_currency}-{against_currency}', period='10y', interval='1d')
eth_data = pd.DataFrame(eth_data)
eth_data['prediction'] = eth_data['Close'].shift(periods=-1)
eth_data['return_prediction'] = eth_data['prediction']/ eth_data['Close'] -1

#JOIN PRICES AND TWEETS
data_merged = pd.merge(data_dtm, eth_data[['return_prediction']], how='left', left_on=['created_at'], right_index = True)
data_merged = data_merged.dropna()
data_merged = data_merged.reset_index(drop=True)
data_merged['action'] = data_merged['return_prediction'].apply(lambda x: 'buy' if x > 0.002 else 'sell')
