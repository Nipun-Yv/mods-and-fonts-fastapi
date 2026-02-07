# db.py
import asyncpg
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

DATABASE_URL=os.getenv("DATABASE_URL")


pool: Optional[asyncpg.Pool] = None

async def init_db():
    global pool
    try:
        pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)
        logger.info(f"Database pool created: {DATABASE_URL}")
    except Exception as e:
        logger.error(f"Failed to create database pool: {e}")
        raise

async def close_db():
    global pool
    if pool:
        await pool.close()
        logger.info("🔒 Database pool closed")

async def get_db():
    if pool is None:
        raise Exception("Database pool not initialized. Call init_db() first.")
    return pool