from services.database.postgres_client import PostgresClient

_db = None

def get_db():
    global _db
    if _db is None:
        _db = PostgresClient()
    return _db

def save_conversation(input_language, input_text, translated_text):
    db = get_db()
    if not db or not db.conn:
        return

    query = """
    INSERT INTO conversations (input_language, input_text, translated_text)
    VALUES (%s, %s, %s)
    """
    db.execute(query, (input_language, input_text, translated_text))
