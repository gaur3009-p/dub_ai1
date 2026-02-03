from services.database.postgres_client import PostgresClient

db = PostgresClient()

def save_conversation(input_language, input_text, translated_text):
    query = """
    INSERT INTO conversations (input_language, input_text, translated_text)
    VALUES (%s, %s, %s)
    """
    db.execute(query, (input_language, input_text, translated_text))
