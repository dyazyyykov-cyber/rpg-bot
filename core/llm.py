# LLM adapter for narrative generation
import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Stub implementations - will be replaced with actual LLM calls

async def generate_general(context: dict) -> str:
    """Generate general narrative for all players"""
    # TODO: Implement actual LLM call
    # Build prompt from context
    prompt = build_prompt_general(context)
    # For now, return stub
    return f"[Narrator] Something happens in the game... (stub)"

async def generate_private(context: dict) -> str:
    """Generate private narrative for specific player"""
    # TODO: Implement actual LLM call
    # Build prompt from context
    prompt = build_prompt_private(context)
    # For now, return stub
    return f"[Private] You sense something... (stub)"

def build_prompt_general(context: dict) -> str:
    """Build prompt for general narrative"""
    # TODO: Implement proper prompt building
    return str(context)

def build_prompt_private(context: dict) -> str:
    """Build prompt for private narrative"""
    # TODO: Implement proper prompt building
    return str(context)
