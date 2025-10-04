#!/usr/bin/env python3
"""Minimal Telegram RPG Bot using aiogram"""
import os
import json
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
import redis.asyncio as redis

from core.db import init_db, get_or_create_player, get_or_create_session, save_event

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bot configuration
TG_TOKEN = os.getenv("TG_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
GROUP_CHAT_ID = os.getenv("GROUP_CHAT_ID")

if not TG_TOKEN:
    raise ValueError("TG_TOKEN environment variable is required")

# Initialize bot and dispatcher
bot = Bot(token=TG_TOKEN)
dp = Dispatcher()

# Redis connection
redis_client = None


async def init_redis():
    """Initialize Redis connection"""
    global redis_client
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Redis connection established")


async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """/start command - register player"""
    try:
        # Get user info
        user_id = message.from_user.id
        user_name = message.from_user.full_name or f"User_{user_id}"
        
        # Register or get player from database
        player = await get_or_create_player(tg_id=user_id, name=user_name)
        
        logger.info(f"Player registered/retrieved: {player.name} (ID: {player.id}, TG: {player.tg_id})")
        
        # Send welcome message
        welcome_text = (
            f"🎮 Welcome, {player.name}!\n\n"
            "You've been registered in the RPG bot.\n\n"
            "Available commands:\n"
            "/start - Register/check your player\n"
            "/act <action> - Perform an action in the game\n\n"
            "Example: /act search for treasure"
        )
        
        await message.reply(welcome_text)
        
    except Exception as e:
        logger.error(f"Error in /start command: {e}", exc_info=True)
        await message.reply("❌ An error occurred during registration. Please try again.")


@dp.message(Command("act"))
async def cmd_act(message: Message):
    """/act command - send action to Redis queue"""
    try:
        # Get user info
        user_id = message.from_user.id
        user_name = message.from_user.full_name or f"User_{user_id}"
        
        # Extract action text
        action_text = message.text.split(maxsplit=1)[1] if len(message.text.split()) > 1 else ""
        
        if not action_text:
            await message.reply("⚠️ Please specify an action.\nExample: /act search for treasure")
            return
        
        # Get or create player
        player = await get_or_create_player(tg_id=user_id, name=user_name)
        
        # Create action payload
        action_payload = {
            "player_id": player.id,
            "player_tg_id": player.tg_id,
            "player_name": player.name,
            "action": action_text,
            "chat_id": message.chat.id,
            "message_id": message.message_id
        }
        
        # Send to Redis queue
        if redis_client:
            await redis_client.lpush("rpg:actions", json.dumps(action_payload))
            logger.info(f"Action queued: {player.name} -> {action_text}")
        
        # Also save to database as event
        session_id = os.getenv("SESSION_ID", "default_session")
        await get_or_create_session(session_id=session_id, group_chat_id=message.chat.id)
        
        event = {
            "actor_id": player.id,
            "type": "action",
            "payload": {
                "action": action_text,
                "chat_id": message.chat.id
            }
        }
        await save_event(session_id=session_id, event=event)
        
        # Confirm action received
        await message.reply(
            f"✅ Action received: *{action_text}*\n\n"
            "Your action has been queued for processing.",
            parse_mode="Markdown"
        )
        
    except IndexError:
        await message.reply("⚠️ Please specify an action.\nExample: /act search for treasure")
    except Exception as e:
        logger.error(f"Error in /act command: {e}", exc_info=True)
        await message.reply("❌ An error occurred while processing your action. Please try again.")


@dp.message()
async def handle_message(message: Message):
    """Handle all other messages"""
    # For now, just log and ignore
    logger.info(f"Unhandled message from {message.from_user.id}: {message.text}")


async def main():
    """Main function to start the bot"""
    try:
        # Initialize database
        logger.info("Initializing database...")
        await init_db()
        
        # Initialize Redis
        logger.info("Initializing Redis...")
        await init_redis()
        
        # Start bot
        logger.info("Starting bot...")
        await dp.start_polling(bot)
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}", exc_info=True)
    finally:
        # Cleanup
        await close_redis()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
