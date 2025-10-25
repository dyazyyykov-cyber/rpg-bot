async def run_turn_with_llm(
    state: Dict[str, Any],
    actions: List[Dict[str, Any]],
    cfg: TurnConfig,
) -> TurnOutputs:
    # Санитизация входных actions
    norm_actions: List[Dict[str, Any]] = []
    for a in actions or []:
        if isinstance(a, dict):
            norm_actions.append({
                "player_id": str(a.get("player_id") or ""),
                "player_name": str(a.get("player_name") or ""),
                "text": str(a.get("text") or ""),
            })
        else:
            norm_actions.append({
                "player_id": str(getattr(a, "player_id", "") or ""),
                "player_name": str(getattr(a, "player_name", "") or ""),
                "text": str(getattr(a, "text") or ""),
            })

    # Try storylet resolution first
    for action in norm_actions:
        storylet_result = propose_storylet_resolution(state, action)
        if storylet_result and storylet_result.get('confidence', 0) > 0.6:
            logger.info('STORYLET: resolved action via %s', storylet_result.get('storylet_id'))
            # Apply effects directly
            apply_effects(state, storylet_result['effects'])
            # Store narratives
            push_raw_story(state, storylet_result['raw'])
            # Return early - no LLM needed for this action
            # (For now, we'll continue through LLM for other actions)
