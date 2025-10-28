# rpg-bot

Telegram RPG bot that narrates cooperative adventures.

## Turn processing overview

Each turn now uses a single LLM request:

1. `core.turn_runner.run_single_turn` waits for player actions and forwards them to the engine.
2. `core.engine.run_turn_with_llm` builds a contextual prompt, calls the model once and stores the
   resulting narration in the session state.
3. The generated story is broadcast via `TurnIOCallbacks.notify_general_story`, and the updated state
   is saved for the next round.

The narrative is produced directly as prose (no intermediate JSON schemas or automatic effect
application). The game state keeps a short history of previous stories to provide context for future
turns.
