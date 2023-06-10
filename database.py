import sqlite3


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to the database: {db_file}")
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn


def create_tables(conn):
    try:
        c = conn.cursor()

        # Tabelle price_predictions erstellen
        c.execute('''CREATE TABLE IF NOT EXISTS price_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        actual_price REAL,
                        prediction_price REAL
                    )''')

        # Tabelle news erstellen
        c.execute('''CREATE TABLE IF NOT EXISTS news (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT,
                        link TEXT,
                        pub_date TEXT
                    )''')

        c.execute('''CREATE TABLE IF NOT EXISTS tweets (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          text_tweet TEXT,
                          author TEXT,
                          created_at TEXT,
                          views INTEGER,
                          length INTEGER
                      )''')

        conn.commit()
        print("Tables created successfully")
    except sqlite3.Error as e:
        print(e)


def insert_price_prediction(conn, actual_price, prediction_price):
    sql = '''INSERT INTO price_predictions(actual_price, prediction_price)
             VALUES(?, ?)'''

    try:
        cur = conn.cursor()
        cur.execute(sql, (actual_price, prediction_price))
        conn.commit()
        print("Price prediction inserted successfully")
    except sqlite3.Error as e:
        print(e)


def insert_news(conn, title, link, pub_date):
    sql = '''INSERT INTO news(title, link, pub_date)
             VALUES(?, ?, ?)'''

    try:
        cur = conn.cursor()
        cur.execute(sql, (title, link, pub_date))
        conn.commit()
        print("News inserted successfully")
    except sqlite3.Error as e:
        print(e)

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

# Funktion zum Überprüfen, ob der Tweet in der Datenbank vorhanden ist
def check_tweet_exists(conn, text_tweet, author, created_at, views, length):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tweets WHERE text_tweet = ? AND author = ? AND created_at = ? AND views = ? AND length = ?",
                   (text_tweet, author, created_at, views, length))
    result = cursor.fetchone()
    cursor.close()
    return result is not None

# Funktion zum Überprüfen, ob die Nachricht in der Datenbank vorhanden ist
def check_news_exists(conn, title, link):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM news WHERE title = ? AND link = ?", (title, link))
    result = cursor.fetchone()
    cursor.close()
    return result is not None