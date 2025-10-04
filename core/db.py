# Database connection and initialization
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from .models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./game.db")

engine = create_async_engine(DATABASE_URL, echo=True)
Async SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    """Get database session"""
    async with AsyncSessionLocal() as session:
        yield session

# Stub functions for session and player management
async def get_session_and_players(session_id: str):
    """Get session state and players - stub"""
    # TODO: Implement actual DB queries
    return {}, []

async def save_event(session_id: str, event: dict):
    """Save event to database - stub"""
    # TODO: Implement actual DB insert
    pass
