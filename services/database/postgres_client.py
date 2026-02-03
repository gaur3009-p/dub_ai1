import psycopg2
from config.settings import POSTGRES

class PostgresClient:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=POSTGRES["host"],
            port=POSTGRES["port"],
            database=POSTGRES["db"],
            user=POSTGRES["user"],
            password=POSTGRES["password"],
            sslmode=POSTGRES.get("sslmode", "require"),
        )
        self.conn.autocommit = True

    def execute(self, query, params=None):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:
                return cur.fetchall()
