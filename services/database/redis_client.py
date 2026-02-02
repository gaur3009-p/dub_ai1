import redis
from config.settings import REDIS

redis_client = redis.Redis(
    host=REDIS["host"],
    port=REDIS["port"],
    db=REDIS["db"],
    decode_responses=True
)
