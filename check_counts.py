from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.database.database import AsyncSessionLocal


async def main() -> None:
    async with AsyncSessionLocal() as session:
        documents_count = (
            await session.execute(text("SELECT COUNT(*) FROM documents"))
        ).scalar_one()
        document_chunks_count = (
            await session.execute(text("SELECT COUNT(*) FROM document_chunks"))
        ).scalar_one()

    print(f"documents: {documents_count}")
    print(f"document_chunks: {document_chunks_count}")


if __name__ == "__main__":
    asyncio.run(main())