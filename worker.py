import os
import json
import asyncio
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from openai import AsyncOpenAI

from core.db import init_db, Base
from core.rules import apply_rules
from core.models import Event


class Worker:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/rpgbot")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        self.engine = None
        self.session_maker = None
        self.redis_client = None
        self.openai_client = None

    async def init_connections(self):
        """Initialize database, Redis and OpenAI connections"""
        # Database
        self.engine = create_async_engine(self.db_url, echo=False)
        await init_db(self.engine)
        self.session_maker = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Redis
        self.redis_client = await aioredis.from_url(self.redis_url)
        
        # OpenAI
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        
        print("Worker initialized: DB, Redis, OpenAI connected")

    async def process_action(self, action_data: dict):
        """Process action: apply rules, save event, generate narrative"""
        async with self.session_maker() as session:
            # Apply rule engine
            result = await apply_rules(action_data, session)
            
            # Save event
            event = Event(
                user_id=action_data["user_id"],
                action_type=action_data["action_type"],
                data=action_data,
                result=result
            )
            session.add(event)
            await session.commit()
            
            # Generate narrative with LLM
            narrative = await self.generate_narrative(action_data, result)
            
            # Send narrative to user (push to response queue)
            await self.send_narrative(action_data["user_id"], narrative)
            
            return narrative

    async def generate_narrative(self, action_data: dict, result: dict) -> str:
        """Call LLM to generate narrative text"""
        prompt = f"""Action: {action_data['action_type']}
Result: {json.dumps(result)}
Generate a short RPG narrative (2-3 sentences)."""
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        return response.choices[0].message.content

    async def send_narrative(self, user_id: int, narrative: str):
        """Push narrative to response queue for bot to send"""
        response_data = {
            "user_id": user_id,
            "narrative": narrative
        }
        await self.redis_client.rpush(
            "response_queue",
            json.dumps(response_data)
        )

    async def run(self):
        """Main worker loop: blpop from action queue and process"""
        await self.init_connections()
        
        print("Worker started, listening for actions...")
        
        while True:
            try:
                # Block until action arrives (blpop with 1 second timeout)
                result = await self.redis_client.blpop("action_queue", timeout=1)
                
                if result:
                    _, action_json = result
                    action_data = json.loads(action_json)
                    
                    print(f"Processing action: {action_data}")
                    await self.process_action(action_data)
                    
            except Exception as e:
                print(f"Error processing action: {e}")
                await asyncio.sleep(1)


if __name__ == "__main__":
    worker = Worker()
    asyncio.run(worker.run())
