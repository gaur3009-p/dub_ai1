import redis
from config.settings import REDIS

redis_client = redis.Redis(
    host=REDIS["host"],
    port=REDIS["port"],
    username=REDIS.get("username"),
    password=REDIS.get("password"),
    db=REDIS.get("db", 0),
    decode_responses=True,
    ssl=REDIS.get("ssl", True),
)
