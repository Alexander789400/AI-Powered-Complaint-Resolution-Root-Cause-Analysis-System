import sqlite3

def get_db_connection():
    conn = sqlite3.connect("complaints.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint_text TEXT,
        transformer_category TEXT,
        lstm_category TEXT,
        sentiment TEXT,
        priority TEXT,
        confidence REAL,
        root_cause TEXT,
        automated_reply TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
