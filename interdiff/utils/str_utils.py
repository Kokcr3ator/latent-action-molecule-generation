def with_bounds(token_string: str) -> str:
    """Add explicit BOS/EOS markers (also space-separated)."""
    return "[BOS] " + token_string + " [EOS]"