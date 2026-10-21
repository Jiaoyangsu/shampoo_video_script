"""Compatibility wrapper for the old structured-form UI.

The old file had a missing ``import os`` and duplicated the whole Ollama call
stack.  The canonical implementation now lives in ``src.app.chat_ui``; this
wrapper keeps the old import path working without maintaining a second app.
"""

from src.app.chat_ui import build_app, main

create_form_demo = build_app

if __name__ == "__main__":
    main()
