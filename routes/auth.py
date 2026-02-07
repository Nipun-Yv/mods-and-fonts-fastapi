from fastapi import HTTPException, Request, APIRouter
import secrets
import logging
import httpx
from typing import Dict
from config.db import get_db
from dotenv import load_dotenv
from pathlib import Path

import os

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

DROPBOX_CLIENT_ID = os.getenv("DROPBOX_CLIENT_ID")
DROPBOX_CLIENT_SECRET =  os.getenv("DROPBOX_CLIENT_SECRET")
DROPBOX_REDIRECT_URI = os.getenv("DROPBOX_REDIRECT_URI")
FRONTEND_REDIRECT_URI =os.getenv("FRONTEND_REDIRECT_URI")


router = APIRouter()
logger = logging.getLogger(__name__)

token_store: Dict[str, Dict] = {}
state_store: Dict[str, str] = {}

from pydantic import BaseModel

class StateRegistration(BaseModel):
    state: str

@router.post("/register-state")
async def register_state(data: StateRegistration):
    state_store[data.state] = "pending"
    logger.info(f"Registered state: {data.state}")
    return {"status": "registered"}


@router.post("/dropbox/exchange")
async def exchange_token(request: Request):
    data = await request.json()
    code = data.get('code')
    state = data.get('state')
    redirect_uri = data.get('redirect_uri')
    code_verifier = data.get('code_verifier')
    
    logger.info(f"Received token exchange request")
    logger.info(f"Code: {code[:10] if code else 'None'}...")
    logger.info(f"State: {state}")
    logger.info(f"Redirect URI: {redirect_uri}")
    logger.info(f"Code Verifier: {code_verifier[:10] if code_verifier else 'None'}...")
    
    if state not in state_store:
        logger.error(f"Invalid state: {state}")
        raise HTTPException(status_code=400, detail="Invalid state")
    
    token_exchange_data = {
        "code": code,
        "grant_type": "authorization_code",
        "client_id": DROPBOX_CLIENT_ID,
        "client_secret": DROPBOX_CLIENT_SECRET,
    }
    
    if redirect_uri:
        token_exchange_data["redirect_uri"] = redirect_uri
    
    if code_verifier:
        token_exchange_data["code_verifier"] = code_verifier
    
    async with httpx.AsyncClient() as client:
        try:
            token_response = await client.post(
                "https://api.dropboxapi.com/oauth2/token",
                data=token_exchange_data
            )
            token_response.raise_for_status()
            token_data = token_response.json()
            
            user_response = await client.post(
                "https://api.dropboxapi.com/2/users/get_current_account",
                headers={
                    "Authorization": f"Bearer {token_data['access_token']}"
                },
                json=None  
            )
            
            logger.info(f"User info response status: {user_response.status_code}")
            logger.info(f"User info response: {user_response.text}")
            
            user_response.raise_for_status()
            user_data = user_response.json()
            
            user_email = user_data.get("email", "unknown")
            user_name = user_data.get("name", {}).get("display_name", "User")
            account_id = user_data.get("account_id", "unknown")
            
            logger.info(f"User: {user_name} ({user_email})")
            logger.info(f"Account ID: {account_id}")
            
            session_id = secrets.token_urlsafe(32)
            
            db = await get_db()
            
            user_id = await db.fetchval('''
                INSERT INTO users (dropbox_email, dropbox_account_id)
                VALUES ($1, $2)
                ON CONFLICT (dropbox_account_id) 
                DO UPDATE SET dropbox_email = EXCLUDED.dropbox_email
                RETURNING id
            ''', user_email, account_id)
            
            session_id = secrets.token_urlsafe(32)
            
            await db.execute('''
                INSERT INTO sessions (id, user_id, refresh_token)
                VALUES ($1, $2, $3)
            ''', session_id, user_id, token_data.get("refresh_token"))
            
            del state_store[state]
            
            
            logger.info(f"Stored tokens for session: {session_id}")
            logger.info(f"Total active sessions: {len(token_store)}")
            
            return {
                "session_id": session_id,
                "access_token": token_data["access_token"],
                "expires_in": token_data.get("expires_in", 14400),
                "user_email": user_email,
                "user_name": user_name
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code}")
            logger.error(f"Response: {e.response.text}")
            raise HTTPException(status_code=400, detail=f"API call failed: {e.response.text}")
        except ValueError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to parse response: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
@router.get("/session")
async def get_session(session_id: str):
    db = await get_db()
    
    session = await db.fetchrow('''
        SELECT s.id, u.dropbox_email, u.generation_count, u.generation_limit
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.id = $1
    ''', session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    await db.execute('UPDATE sessions SET last_used_at = CURRENT_TIMESTAMP WHERE id = $1', session_id)
    
    return {
        "session_id": session_id,
        "user_email": session['dropbox_email'],
        "authenticated": True,
        "generation_count": session['generation_count'],
        "generation_limit": session['generation_limit']
    }


@router.post("/logout")
async def logout(session_id: str):
    """
    Logout and revoke tokens
    """
    if session_id in token_store:
        access_token = token_store[session_id]["access_token"]
        
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    "https://api.dropboxapi.com/2/auth/token/revoke",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                logger.info(f"🔒 Token revoked for session: {session_id}")
            except Exception as e:
                logger.error(f"Token revocation failed: {e}")
        
        del token_store[session_id]
        logger.info(f"Logged out session: {session_id}")
    
    return {"message": "Logged out successfully"}

# @app.get("/api/dropbox/files")
# async def list_files(session_id: str, path: str = ""):
#     """
#     Example: List Dropbox files using stored access token
#     """
#     if session_id not in token_store:
#         raise HTTPException(status_code=401, detail="Unauthorized")
    
#     access_token = token_store[session_id]["access_token"]
    
#     async with httpx.AsyncClient() as client:
#         try:
#             response = await client.post(
#                 "https://api.dropboxapi.com/2/files/list_folder",
#                 headers={
#                     "Authorization": f"Bearer {access_token}",
#                     "Content-Type": "application/json"
#                 },
#                 json={"path": path or ""}
#             )
#             response.raise_for_status()
#             return response.json()
#         except httpx.HTTPStatusError as e:
#             logger.error(f"Failed to list files: {e.response.text}")
#             raise HTTPException(status_code=e.response.status_code, detail="Failed to fetch files")

