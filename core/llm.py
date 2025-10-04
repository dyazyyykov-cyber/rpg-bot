# LLM adapter for narrative generation
import os
from openai import AsyncOpenAI

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def generate_general(context: dict) -> str:
    """Generate general narrative for all players using OpenAI ChatCompletion"""
    prompt = build_prompt_general(context)
    
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative RPG game master narrating events for all players."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error generating narrative: {str(e)}]"

async def generate_private(context: dict) -> str:
    """Generate private narrative for specific player using OpenAI ChatCompletion"""
    prompt = build_prompt_private(context)
    
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative RPG game master providing private information to a specific player."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error generating private narrative: {str(e)}]"

def build_prompt_general(context: dict) -> str:
    """Build prompt for general narrative"""
    # Extract relevant context information
    location = context.get('location', 'unknown location')
    event = context.get('event', 'something happens')
    players = context.get('players', [])
    
    prompt = f"""Generate a short RPG narrative for the following situation:
Location: {location}
Event: {event}
Players present: {', '.join(players) if players else 'none'}

Create an engaging narrative that all players can see."""
    return prompt

def build_prompt_private(context: dict) -> str:
    """Build prompt for private narrative"""
    # Extract relevant context information
    player_name = context.get('player_name', 'Player')
    secret_info = context.get('secret_info', 'something mysterious')
    location = context.get('location', 'unknown location')
    
    prompt = f"""Generate a short private RPG narrative for a specific player:
Player: {player_name}
Location: {location}
Secret information: {secret_info}

Create a personal narrative that only this player should see."""
    return prompt
