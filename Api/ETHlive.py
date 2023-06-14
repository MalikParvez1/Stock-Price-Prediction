import requests
import csv
import time
from datetime import datetime

def get_ethereum_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=eur"
    response = requests.get(url)
    data = response.json()
    return data['ethereum']['eur']

def save_price_to_csv(price):
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    date = now.strftime('%Y-%m-%d')
    with open('../Api/eth_price.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([date, timestamp, price])

def main():
    while True:
        try:
            price = get_ethereum_price()
            save_price_to_csv(price)
            print(f"Price saved: {price}")
            time.sleep(60)  # Warte 60 Sekunden
        except Exception as e:
            print(f"Fehler beim Abrufen und Speichern des Preises: {e}")
            time.sleep(60)  # Bei einem Fehler warte 60 Sekunden und versuche es erneut

if __name__ == '__main__':
    main()

