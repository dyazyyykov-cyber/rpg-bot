await redis.rpush(f"session:{session_id}:actions", json.dumps(payload, ensure_ascii=False))
