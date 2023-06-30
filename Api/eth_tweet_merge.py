import sqlite3
import pandas as pd


#Pull the Tweets from the DB

tweet_dataFrame = pd

def fetch_tweets_from_db(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Daten aus der Tabelle "tweets" abrufen
    cursor.execute("SELECT * FROM tweets")
    rows = cursor.fetchall()

    # Spaltennamen abrufen
    columns = [description[0] for description in cursor.description]

    # DataFrame erstellen
    tweets_df = pd.DataFrame(rows, columns=columns)

    # Verbindung zur Datenbank schließen
    cursor.close()
    conn.close()

    return tweets_df


#CSV insert DB




# Datenbankdatei für die Tweets
db_file = "test2.db"

# DataFrame für die Tweets abrufen
tweets_df = fetch_tweets_from_db(db_file)

# DataFrame aus der CSV-Datei für ETH
eth_df = pd.read_csv('C:\\Users\\jpnai\\Uni\\projctUsws\\twitterbot\\Api\\ETHUSD_1.csv')

# Merge der DataFrames
merged_df = pd.merge(tweets_df, eth_df, on="date", how="inner")

# Ergebnis anzeigen
print(merged_df)






