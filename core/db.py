# Database connection and initialization
import os
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select
from .models import Base, Player, Session, Event, Item, Inventory

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./game.db")
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialized successfully")

async def get_db():
    """Get database session"""
    async with AsyncSessionLocal() as session:
        yield session

async def get_session_and_players(session_id: str):
    """Get session state and players
    
    Returns:
        tuple: (session_state dict, list of player objects)
    """
    async with AsyncSessionLocal() as db:
        # Get session
        result = await db.execute(
            select(Session).where(Session.id == session_id)
        )
        session = result.scalar_one_or_none()
        
        if not session:
            # Return empty state if session doesn't exist
            return {"session_id": session_id, "exists": False}, []
        
        # Get all events for this session to rebuild state
        events_result = await db.execute(
            select(Event)
            .where(Event.session_id == session_id)
            .order_by(Event.created_at)
        )
        events = events_result.scalars().all()
        
        # Get all players (simplified - get all for MVP)
        players_result = await db.execute(select(Player))
        players = players_result.scalars().all()
        
        # Build session state from events
        session_state = {
            "session_id": session_id,
            "title": session.title,
            "group_chat_id": session.group_chat_id,
            "exists": True,
            "event_count": len(events),
            "last_events": [
                {
                    "id": e.id,
                    "type": e.type,
                    "payload": e.payload,
                    "actor_id": e.actor_id,
                    "created_at": e.created_at.isoformat() if e.created_at else None
                }
                for e in events[-10:]  # Last 10 events for context
            ]
        }
        
        return session_state, list(players)

async def save_event(session_id: str, event: dict):
    """Save event to database
    
    Args:
        session_id: ID of the session
        event: Event dictionary with keys: type, payload, actor_id (optional)
    """
    async with AsyncSessionLocal() as db:
        # Create event object
        new_event = Event(
            session_id=session_id,
            actor_id=event.get("actor_id"),
            type=event.get("type", "unknown"),
            payload=event.get("payload", {})
        )
        
        db.add(new_event)
        await db.commit()
        await db.refresh(new_event)
        
        return new_event

async def get_or_create_player(tg_id: int, name: str = None):
    """Get existing player or create new one
    
    Args:
        tg_id: Telegram user ID
        name: Player name (optional)
    
    Returns:
        Player object
    """
    async with AsyncSessionLocal() as db:
        # Try to find existing player
        result = await db.execute(
            select(Player).where(Player.tg_id == tg_id)
        )
        player = result.scalar_one_or_none()
        
        if not player:
            # Create new player
            player = Player(
                tg_id=tg_id,
                name=name or f"Player_{tg_id}"
            )
            db.add(player)
            await db.commit()
            await db.refresh(player)
        
        return player

async def get_or_create_session(session_id: str, title: str = None, group_chat_id: int = None):
    """Get existing session or create new one
    
    Args:
        session_id: Session ID
        title: Session title (optional)
        group_chat_id: Group chat ID (optional)
    
    Returns:
        Session object
    """
    async with AsyncSessionLocal() as db:
        # Try to find existing session
        result = await db.execute(
            select(Session).where(Session.id == session_id)
        )
        session = result.scalar_one_or_none()
        
        if not session:
            # Create new session
            session = Session(
                id=session_id,
                title=title or f"Session_{session_id}",
                group_chat_id=group_chat_id
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)
        
        return session
