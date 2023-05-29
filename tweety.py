from tweety.bot import Twitter
from typing import List
from tweety.types import Tweet
import pandas as pd
from datetime import datetime

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
            "date": str(tweet.date.date()),
            "created_at": tweet.date,
            "views": tweet.views,
            "length": len(tweet.text)
        })
    df = pd.DataFrame(data, columns=["id", "text", "author", "date", "views", "created_at"])
    df.set_index("id", inplace=True)
    if df.empty:
        return df
    df = df[df.created_at.dt.date > datetime.now().date() - pd.to_timedelta("7day")]
    return df

all_tweets = app.get_tweets("elonmusk")
print(create_dataframe_from_tweets(all_tweets))
