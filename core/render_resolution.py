"""1K resolution cap for Hermes framebuffers/windows."""

MAX_RENDER_SIDE = 1024


def resolve_max_render_side(value=None):
    """Allow a lower cap, never a ceiling greater than 1024 pixels."""
    try:
        return max(256, min(MAX_RENDER_SIDE, int(value)))
    except (TypeError, ValueError):
        return MAX_RENDER_SIDE


def limit_dimensions(width, height, max_side=MAX_RENDER_SIDE):
    """Keep aspect ratio when restoring oversized window dimensions."""
    width, height = max(1, int(width)), max(1, int(height))
    if max(width, height) <= max_side:
        return width, height
    factor = max_side / max(width, height)
    return max(1, round(width * factor)), max(1, round(height * factor))
