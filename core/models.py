# SQLAlchemy models for RPG bot
from sqlalchemy import Column, Integer, String, JSON, Boolean, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime
import uuid

Base = declarative_base()

def gen_uuid():
    return str(uuid.uuid4())

class Player(Base):
    __tablename__ = "players"
    id = Column(String, primary_key=True, default=gen_uuid)
    tg_id = Column(Integer, unique=True, index=True)
    name = Column(String)

class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=gen_uuid)
    title = Column(String)
    group_chat_id = Column(Integer)

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    actor_id = Column(String, ForeignKey("players.id"), nullable=True)
    type = Column(String)
    payload = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Item(Base):
    __tablename__ = "items"
    id = Column(String, primary_key=True, default=gen_uuid)
    name = Column(String)
    props = Column(JSON)

class Inventory(Base):
    __tablename__ = "inventory"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String, ForeignKey("players.id"))
    item_id = Column(String, ForeignKey("items.id"))
    quantity = Column(Integer, default=1)
