# Utility functions for locks, idempotency, and helpers
import asyncio
import redis.asyncio as aioredis
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Redis lock helper using SETNX + TTL
# IMPORTANT: Idempotency is critical for distributed systems
# Always check if an action has been processed before executing it again

async def acquire_redis_lock(redis_client, lock_key: str, ttl: int = 30) -> bool:
    """Acquire a distributed lock using Redis SETNX with TTL
    
    Args:
        redis_client: Redis client instance
        lock_key: Unique key for the lock
        ttl: Time-to-live in seconds (default: 30)
    
    Returns:
        bool: True if lock acquired, False otherwise
    """
    result = await redis_client.set(lock_key, "1", nx=True, ex=ttl)
    return result is not None

async def release_redis_lock(redis_client, lock_key: str):
    """Release a distributed lock
    
    Args:
        redis_client: Redis client instance
        lock_key: Unique key for the lock
    """
    await redis_client.delete(lock_key)

async def acquire_session_lock(session_id: str, ttl: int = 30) -> bool:
    """Acquire a lock for a session using Redis SETNX"""
    redis_client = aioredis.from_url(REDIS_URL)
    lock_key = f"session:lock:{session_id}"
    return await acquire_redis_lock(redis_client, lock_key, ttl)

async def release_session_lock(session_id: str):
    """Release a session lock"""
    redis_client = aioredis.from_url(REDIS_URL)
    lock_key = f"session:lock:{session_id}"
    await release_redis_lock(redis_client, lock_key)

async def check_action_processed(action_id: str) -> bool:
    """Check if an action has already been processed (idempotency)"""
    redis_client = aioredis.from_url(REDIS_URL)
    key = f"action:processed:{action_id}"
    exists = await redis_client.exists(key)
    return exists > 0

async def mark_action_processed(action_id: str, ttl: int = 3600):
    """Mark an action as processed"""
    redis_client = aioredis.from_url(REDIS_URL)
    key = f"action:processed:{action_id}"
    await redis_client.set(key, "1", ex=ttl)
