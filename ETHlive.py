import requests
import csv
import time
from datetime import datetime
from database import create_tables, create_connection, insert_price_prediction, check_price_prediction_exists

def get_ethereum_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data['ethereum']['usd']

def save_price_to_csv(price):
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    date = now.strftime('%Y-%m-%d')
    with open('Api/eth_price.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([date, timestamp, price])

def insert_newest_price_to_database(conn):
    with open('Api/eth_price.csv', 'r') as file:
        csv_reader = csv.reader(file)
        rows = list(csv_reader)
        if len(rows) > 1:
            latest_row = rows[-1]
            date_time = latest_row[1]
            actual_price = float(latest_row[2])
            predicted_price = None

            if len(latest_row) >= 4:
                predicted_price = float(latest_row[3])

            insert_price_prediction(conn, date_time, actual_price, predicted_price)

            print("Newest data inserted into the database successfully")
        else:
            print("CSV file does not contain any new data")

def main():
    db_file = "test2.db"  # Specify the path to your database file
    conn = create_connection(db_file)
    create_tables(conn)

    while True:
        try:
            price = get_ethereum_price()
            save_price_to_csv(price)
            insert_newest_price_to_database(conn)
            print(f"Price saved: {price}")
            time.sleep(60)  # Wait for 60 seconds
        except Exception as e:
            print(f"Error retrieving and saving the price: {e}")
            time.sleep(60)  # In case of an error, wait for 60 seconds and try again

    conn.close()

if __name__ == '__main__':
    main()
