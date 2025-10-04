# add minimal rule engine
# Rule engine for game logic
import random
import re

def roll(d=20):
    """Roll a dice with d sides"""
    return random.randint(1, d)

# Primitive action parser
def parse_action_text(text: str) -> dict:
    """Parse user action text into structured action"""
    text = text.lower().strip()
    
    # Attack patterns
    if re.search(r'\b(attack|атак|удар|бить)\b', text):
        target_match = re.search(r'\b(?:attack|атак|удар|бить)\s+(\w+)', text)
        target = target_match.group(1) if target_match else "unknown"
        return {"type": "attack", "target": target}
    
    # Search patterns
    if re.search(r'\b(search|искать|обыск|осмотр)\b', text):
        location_match = re.search(r'\b(?:search|искать|обыск|осмотр)\s+(\w+)', text)
        location = location_match.group(1) if location_match else "room"
        return {"type": "search", "location": location}
    
    # Pickup patterns
    if re.search(r'\b(pickup|pick up|взять|подобрать|grab|take)\b', text):
        item_match = re.search(r'\b(?:pickup|pick up|взять|подобрать|grab|take)\s+(\w+)', text)
        item = item_match.group(1) if item_match else "item"
        return {"type": "pickup", "item": item}
    
    # Default: generic action
    return {"type": "custom", "text": text}

# Attack handler
async def handle_attack(state, actor, target):
    """Handle attack action"""
    atk_roll = roll(20)
    success = atk_roll + actor.get('str_mod', 0) >= target.get('ac', 10)
    
    if success:
        dmg = random.randint(1, 8) + actor.get('str_mod', 0)
        target['hp'] = target.get('hp', 10) - dmg
        return {
            "type": "attack",
            "actor": actor['id'],
            "target": target['id'],
            "success": True,
            "damage": dmg,
            "roll": atk_roll,
            "target_hp": target['hp']
        }
    else:
        return {
            "type": "attack",
            "actor": actor['id'],
            "target": target['id'],
            "success": False,
            "roll": atk_roll
        }

# Search handler
async def handle_search(state, actor, location: str):
    """Handle search action"""
    perception_roll = roll(20)
    dc = state.get('search_dc', 12)
    success = perception_roll + actor.get('perception_mod', 0) >= dc
    
    if success:
        # Found something
        found_items = state.get('hidden_items', [])
        found_item = found_items[0] if found_items else None
        return {
            "type": "search",
            "actor": actor['id'],
            "location": location,
            "success": True,
            "roll": perception_roll,
            "found_item": found_item
        }
    else:
        return {
            "type": "search",
            "actor": actor['id'],
            "location": location,
            "success": False,
            "roll": perception_roll
        }

# Pickup handler
async def handle_pickup(state, actor, item_name: str):
    """Handle pickup action"""
    available_items = state.get('available_items', [])
    
    # Check if item exists
    item = next((i for i in available_items if i.get('name', '').lower() == item_name.lower()), None)
    
    if item:
        return {
            "type": "pickup",
            "actor": actor['id'],
            "item": item,
            "success": True
        }
    else:
        return {
            "type": "pickup",
            "actor": actor['id'],
            "item_name": item_name,
            "success": False
        }

# Validation
async def validate_action(session, player, action):
    """Validate if action is allowed"""
    # Basic validation
    if not action or not action.get('type'):
        return False
    
    # Check if player is alive
    if player.get('hp', 0) <= 0:
        return False
    
    # Check if it's player's turn (optional for MVP)
    # Add more sophisticated validation as needed
    
    return True

# Main resolution function
async def resolve_action(session, action_set):
    """Resolve a set of actions into events"""
    events = []
    state = session.get('state', {})
    players = session.get('players', {})
    npcs = state.get('npcs', [])
    
    for action in action_set:
        actor = players.get(action['player_id'])
        if not actor:
            continue
        
        # Validate action
        if not await validate_action(session, actor, action):
            events.append({
                "type": "invalid_action",
                "actor": action['player_id'],
                "reason": "Action not allowed"
            })
            continue
        
        # Parse and route action
        parsed = action.get('parsed', {})
        action_type = parsed.get('type')
        
        if action_type == 'attack':
            # Find target (NPC or another player)
            target_name = parsed.get('target')
            target = next((npc for npc in npcs if npc.get('name', '').lower() == target_name.lower()), None)
            
            if target:
                event = await handle_attack(state, actor, target)
                events.append(event)
            else:
                events.append({
                    "type": "attack",
                    "actor": actor['id'],
                    "success": False,
                    "reason": "Target not found"
                })
        
        elif action_type == 'search':
            location = parsed.get('location', 'room')
            event = await handle_search(state, actor, location)
            events.append(event)
        
        elif action_type == 'pickup':
            item_name = parsed.get('item')
            event = await handle_pickup(state, actor, item_name)
            events.append(event)
        
        else:
            # Generic custom action
            events.append({
                "type": "custom",
                "actor": actor['id'],
                "text": parsed.get('text', '')
            })
    
    return events
