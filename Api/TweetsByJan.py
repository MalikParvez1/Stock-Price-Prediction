import pandas as pd
import snscrape.modules.twitter as sntwitter
from tqdm import tqdm




#Start at begin
tweets_n = 5000

query = "(Increase, OR deincrease, OR decline, OR jump, OR take OR off) (#Krypto, OR #Bitcoin) until:2017-06-29 since:2017-01-01"

#for Elon
#query = "bitcoin (from:elonmusk) until:2022-11-30 since:2017-01-01"


scraper = sntwitter.TwitterSearchScraper(query)
#Pull the tweets
for tweet in scraper.get_items():
    break




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

#Data for Bitcoin all
tweets = []

for i, tweet in enumerate(scraper.get_items()):
    data = [
        tweet.date,
        tweet.content,
        tweet.user.username,
        ]
    tweets.append(data)
    if i > 5000:
        break
tweet_df = pd.DataFrame(
    tweets, columns=["date", "content", "username"]
)

tweet_df.to_csv("tweets-Bitcoin", index=False)





#Add Bar func dosen't work

tweets = []

for i, tweet in tqdm(enumerate(scraper.get_items()), total=1000):

    data = [
            tweet.date,
            tweet.content,
            tweet.user.username,
    ]
    tweets.append(data)

    if i >= 1000:
        break

tweet_df = pd.DataFrame(
    tweets, columns=["date", "content", "username"]
)

tweet_df.to_csv("tweets-Jan-Test", index=False)





#Tweets Elon Músk
tweets = []

for i, tweet in enumerate(scraper.get_items()):
    data = [
        tweet.date,
        tweet.content,
        tweet.user.username,
    ]
    tweets.append(data)
    if i > tweets_n:
        break
tweet_df = pd.DataFrame(
    tweets, columns=["date", "content", "username"]
)

tweet_df.to_csv("tweets-Elon_Musk_Bitcoin", index=False)



