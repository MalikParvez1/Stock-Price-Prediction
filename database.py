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

        # Table price_predictions
        c.execute('''CREATE TABLE IF NOT EXISTS price_predictions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                date_time DATETIME,
                                actual_price REAL,
                                predicted_price REAL
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


def insert_price_prediction(conn, date_time, actual_price, predicted_price):
    sql = '''INSERT INTO price_predictions(date_time, actual_price, predicted_price)
             VALUES(?, ?, ?)'''

    try:
        cur = conn.cursor()
        cur.execute(sql, (date_time, actual_price, predicted_price))
        conn.commit()
        print("Price prediction inserted successfully")
    except sqlite3.Error as e:
        print(e)

# Funktion zum Überprüfen, ob die Preisvorhersage in der Datenbank vorhanden ist
def check_price_prediction_exists(conn, actual_price, prediction_price):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM price_predictions WHERE actual_price = ? AND predicted_price = ?",
                   (actual_price, prediction_price))
    result = cursor.fetchone()
    cursor.close()
    return result is not None


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
    cursor.execute("SELECT * FROM tweets WHERE text_tweet = ? AND author = ? AND created_at = ?",
                   (text_tweet, author, created_at))
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