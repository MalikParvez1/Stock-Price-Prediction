import pandas as pd
import os

df = pd.read_csv('../ETHUSD_1.csv')


df.rename(columns={"1438956180": "Date", "3.0": "Open","3.0.1": "High", "3.0.2": "Low", "3.0.3": "Close", "81.85727776": "Volume", "2": "Trades"}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')

MAX_QUOTE_ASSET_VOLUME = df.loc[
    df['Volume'].idxmax()]['Volume']

MAX_NUMBER_of_TRADES = df.loc[
    df['Trades'].idxmax()]['Trades']


MAX_ACCOUNT_BALANCE = 10000000
INITIAL_ACCOUNT_BALANCE = 1000
MAX_CRYPTO_PRICE = 20000
MAX_CRYPTO = 21000000
MAKER_FEE = 0.00075
TAKER_FEE = 0.00075
BNBUSDTHELD = 1000
MAX_STEPS = int(os.getenv('MAX_STEPS', 400))