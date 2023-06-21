from database import create_connection, create_tables, insert_news, check_news_exists
import requests
from bs4 import BeautifulSoup
import time

crypto_currency = 'BTC'

def scrape_google_news(query):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    articles = soup.findAll('item')
    news = []

    for article in articles:
        title = article.title.text
        link = article.link.text
        pub_date = article.pubDate.text
        news.append({
            'title': title,
            'link': link,
            'pub_date': pub_date
        })

    return news

# Verbindung zur Datenbank herstellen
conn = create_connection("test2.db")

# Tabellen erstellen
create_tables(conn)

while True:
    # Suchanfrage für die Google News
    query = f'{crypto_currency} news'  # Query to search for news

    # News von Google News abrufen
    news_headlines = scrape_google_news(query)

    # Neue News in die Datenbank speichern
    for headline in news_headlines:
        title = headline['title']
        link = headline['link']
        pub_date = headline['pub_date']

        # Überprüfen, ob die Nachricht bereits in der Datenbank vorhanden ist
        if not check_news_exists(conn, title, link):
            # Rufe die Funktion insert_news auf, um die Daten in der Datenbank zu speichern
            insert_news(conn, title, link, pub_date)

    # Wartezeit von 1 Minute
    time.sleep(60)

# Verbindung zur Datenbank schließen
conn.close()