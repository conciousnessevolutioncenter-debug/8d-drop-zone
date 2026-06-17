"""Social layer for The 8D Engine — accounts, profiles, feed, and (later) chat.

Self-contained and additive: the audio masterer keeps working with or without
this package. Everything is mounted under /social so the existing routes are
untouched. Local dev uses SQLite; production uses Postgres via DATABASE_URL.
"""
