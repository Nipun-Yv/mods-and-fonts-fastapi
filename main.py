from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from fastapi.middleware.cors import CORSMiddleware
from routes.auth import router as auth_router
from routes.generative import router as gen_router
from routes.fonts import router as font_router

from config.db import init_db, close_db


from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Design Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    await close_db()
    logger.info("Database closed")


app.include_router(router=auth_router,prefix="/api/auth")
app.include_router(router=gen_router, prefix="/generative")
app.include_router(router=font_router,prefix="/font")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
