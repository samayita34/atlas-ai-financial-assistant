import asyncio
from sqlalchemy import text
from app.database.database import AsyncSessionLocal

async def main():
    async with AsyncSessionLocal() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await session.commit()
        print("pgvector enabled successfully.")

if __name__ == "__main__":
    asyncio.run(main())
