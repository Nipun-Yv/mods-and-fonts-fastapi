from fastapi.exceptions import HTTPException
from config.db import get_db
async def get_user_from_session(session_id: str):
    db = await get_db()
    
    session = await db.fetchrow('''
        SELECT u.*
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.id = $1
    ''', session_id)
    
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    return dict(session)