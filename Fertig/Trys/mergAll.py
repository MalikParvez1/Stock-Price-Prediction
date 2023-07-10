from datetime import datetime

import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec


#ETH

# CSV-Datei einlesen und nur ausgewählte Spalten laden
selected_columns = [0, 5, 6]
eth_df = pd.read_csv('/twitterbot/Stock-Price-Prediction/Api/ETHUSD_1.csv', usecols=selected_columns, header=None)

# Spaltennamen aktualisieren
eth_df.columns = ['Date', 'Price_ETH', 'Investor_ETH']

# Timestamp in ein Datum umwandeln
eth_df['Date'] = pd.to_datetime(eth_df['Date'], unit='s')

# Datum in das neue Format umwandeln
eth_df['Date'] = eth_df['Date'].apply(lambda x: x.strftime("%m/%d/%Y %H:%M"))

# Ausgabe des DataFrames mit dem formatierten Datum
print(eth_df)

#Bitcoin

# CSV-Datei einlesen und nur ausgewählte Spalten laden
selected_columns = [0, 5, 6]
bitcoin_df = pd.read_csv('/twitterbot/Stock-Price-Prediction/Api/BITUSD_1.csv', usecols=selected_columns, header=None)

# Spaltennamen aktualisieren
bitcoin_df.columns = ['Date', 'Price_BitCoin', 'Investor_BitCoin']

# Timestamp in ein Datum umwandeln
bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'], unit='s')

# Datum in das neue Format umwandeln
bitcoin_df['Date'] = bitcoin_df['Date'].apply(lambda x: x.strftime("%m/%d/%Y %H:%M"))

# CSV-Datei mit den ausgewählten Daten und der ID speichern
bitcoin_df.to_csv('bitcoin_data.csv', index=False)

print(bitcoin_df)



#tweets

# CSV-Datei einlesen und gewünschte Spalten auswählen
tweets_csv_df= pd.read_csv('/twitterbot/Stock-Price-Prediction/Api/Tweets data on Cryptocurrency.csv', usecols=['user_name', 'date', 'user_followers', 'text'])


# DataFrame anzeigen
print(tweets_csv_df)

# Spalte für Textlänge hinzufügen
tweets_csv_df['text_length'] = tweets_csv_df['text'].apply(len)

# ID-Spalte hinzufügen
tweets_csv_df['id'] = range(1, len(tweets_csv_df) + 1)

# DataFrame anzeigen
print(tweets_csv_df)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'http\S+|www\S+|@\S+', '', cleaned_text)
    return cleaned_text

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

# CSV-Datei einlesen und gewünschte Spalten auswählen
tweets_csv_df = pd.read_csv('/twitterbot/Stock-Price-Prediction/Api/Tweets data on Cryptocurrency.csv', usecols=['user_name', 'date', 'user_followers', 'text'])

# Spalte für Textlänge hinzufügen
tweets_csv_df['text_length'] = tweets_csv_df['text'].apply(len)

# Textbereinigung
tweets_csv_df['cleaned_content'] = tweets_csv_df['text'].apply(clean_text)

# Tokenisierung und Textnormalisierung
tweets_csv_df['preprocessed_text'] = tweets_csv_df['cleaned_content'].apply(preprocess_text)

# Stopwortentfernung
tweets_csv_df['filtered_text'] = tweets_csv_df['preprocessed_text'].apply(remove_stopwords)

# Word Embeddings - Beispiel mit Word2Vec
word2vec_model = Word2Vec(tweets_csv_df['filtered_text'], min_count=1)

print(tweets_csv_df)


# DataFrames anhand des gemeinsamen Datums (Spalte "Date") zusammenführen
merged_df = pd.merge(bitcoin_df, eth_df, on='Date', how='inner')
merged_df = pd.merge(merged_df, tweets_csv_df, left_on='Date', right_on='date', how='left')

# Ergebnis anzeigen
print(merged_df)