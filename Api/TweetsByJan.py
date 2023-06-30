from socket import create_connection

import pandas as pd
import snscrape.modules.twitter as sntwitter
from tqdm import tqdm
import sqlite3



#Start at begin


tweets_n = 5000

query = "(Increase, OR deincrease, OR decline, OR jump, OR take OR off) (#Krypto, OR #Bitcoin) until:2017-06-29 since:2017-01-01"

#for Elon
#query = "bitcoin (from:elonmusk) until:2022-11-30 since:2017-01-01"


scraper = sntwitter.TwitterSearchScraper(query)
#Pull the tweets
for tweet in scraper.get_items():
    break

#ende of the first carry out



#type of Tweets
type(tweet)

#example tweet
tweet

#example tweet in a List with the most importend infromation
data = [
    tweet.date,
    tweet.content,
    tweet.user.username,
    tweet.likeCount,
    tweet.retweetedTweet
]





#Codes for pulling








#Tweets fuktioniert

tweets = []

for i, tweet in enumerate(scraper.get_items()):
    data = [
        tweet.id,
        tweet.rawContent,
        tweet.user.username,
        tweet.date,
        tweet.viewCount,
        len(tweet.rawContent)
    ]
    tweets.append(data)
    if i > tweets_n:
        break
tweet_df = pd.DataFrame(
    tweets, columns=["id", "content", "username", "date", "views", "length"]
)

tweet_df


#Data insert in to Tweet


def insert_tweet(conn, text_tweet, author, created_at, views, length):
    sql = '''INSERT INTO tweets(text_tweet, author, created_at, views, length)
             VALUES(?, ?, ?, ?, ?)'''

    try:
        cur = conn.cursor()
        cur.execute(sql, (text_tweet, author, created_at, views, length))
        conn.commit()
        print("Tweet inserted successfully")
    except sqlite3.Error as e:
        print(e)

# Verbindung zur Datenbank herstellen
conn = sqlite3.connect("test2.db")

# DataFrame in die Datenbank einfügen
for _, row in tweet_df.iterrows():
    insert_tweet(conn, row["content"], row["username"], row["date"], row["views"], row["length"])

# Verbindung zur Datenbank schließen
conn.close()




#Not working Code, but good options

#Add Bar func dosen't work

tweets = []

for i, tweet in tqdm(enumerate(scraper.get_items()), total=1000):

    data = [
            tweet.date,
            tweet.content,
            tweet.user.username,
    ]
    tweets.append(data)

    if i >= tweets_n:
        break

tweet_df = pd.DataFrame(
    tweets, columns=["date", "content", "username"]
)

tweet_df.to_csv("tweets-Jan-Test", index=False)
