from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.database.database import AsyncSessionLocal

# Scoped intentionally to exactly these two tables plus the extension
# bootstrap. Does not touch users, conversation_messages, daily_brief_log,
# or watchlist_items.
STATEMENTS: list[str] = [
    "DROP TABLE IF EXISTS document_chunks CASCADE",
    "DROP TABLE IF EXISTS documents CASCADE",
    "CREATE EXTENSION IF NOT EXISTS vector",
]


async def main() -> None:
    print("Starting scoped migration: documents / document_chunks / vector extension")
    print("Statements to execute:")
    for statement in STATEMENTS:
        print(f"  - {statement};")
    print()

    try:
        async with AsyncSessionLocal() as session:
            for statement in STATEMENTS:
                print(f"Executing: {statement};")
                await session.execute(text(statement))
            await session.commit()
    except SQLAlchemyError as exc:
        print()
        print("FAILED — transaction was not committed. Nothing else was touched.")
        print(f"Error: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        print()
        print("FAILED (unexpected error) — transaction was not committed.")
        print(f"Error: {exc}")
        return

    print()
    print("SUCCESS — all three statements committed.")
    print("documents and document_chunks tables dropped.")
    print("vector extension is present (created if it wasn't already).")
    print()
    print("Next step: restart the app so init_db() recreates both tables")
    print("from the corrected models (Document.user_id UUID FK, "
          "DocumentChunk.embedding as Vector(768)).")


if __name__ == "__main__":
    asyncio.run(main())