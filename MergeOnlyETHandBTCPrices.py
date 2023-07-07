import pandas as pd
import pytz

def mergeDataframesPrices():
    berlin_tz = pytz.timezone('Europe/Berlin')

    # TEST DATA NOT IN DATABASE YET - ETH-USD prices
    dfPrices = pd.read_csv('Abgabe/ETHUSD_1.csv')
    dfPrices = dfPrices.dropna()
    dfPrices.rename(columns={"1438956180": "Date", "3.0": "Open", "3.0.1": "High", "3.0.2": "Low", "3.0.3": "Close", "81.85727776": "Volume", "2": "Trades"}, inplace=True)
    dfPrices['Date'] = pd.to_datetime(dfPrices['Date'], unit='s')
    dfPrices['Date'] = pd.to_datetime(dfPrices['Date'], utc=True)
    dfPrices['Date'] = dfPrices['Date'].dt.floor('T')
    dfPrices['Date'] = dfPrices['Date'].dt.tz_convert(berlin_tz)

    dfBtcPrices = pd.read_csv('Abgabe/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')
    dfBtcPrices = dfBtcPrices.dropna()
    dfBtcPrices['Timestamp'] = pd.to_datetime(dfBtcPrices['Timestamp'], unit='s')
    dfBtcPrices['Timestamp'] = pd.to_datetime(dfBtcPrices['Timestamp'], utc=True)
    dfBtcPrices['Timestamp'] = dfBtcPrices['Timestamp'].dt.floor('T')
    dfBtcPrices['Timestamp'] = dfBtcPrices['Timestamp'].dt.tz_convert(berlin_tz)
    dfBtcPrices.rename(columns={"Close": "BTC_Close"}, inplace=True)
    dfBtcPrices = dfBtcPrices.dropna()

    merged_df = pd.merge(dfPrices, dfBtcPrices, left_on='Date', right_on='Timestamp')

    return merged_df

mergeDataframesPrices()
