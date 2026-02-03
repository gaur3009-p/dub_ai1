import psycopg2
from config.settings import POSTGRES, POSTGRES_ENABLED

class PostgresClient:
    def __init__(self):
        if not POSTGRES_ENABLED:
            self.conn = None
            print("⚠️ Postgres disabled: NEON_DATABASE_URL not set")
            return

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
        if not self.conn:
            return None

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:
                return cur.fetchall()
