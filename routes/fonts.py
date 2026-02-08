from fastapi import File, UploadFile, HTTPException, Form,APIRouter, Header
from fastapi.responses import JSONResponse
from utils.font_matcher import match_font, get_available_characters, get_library_info
from utils.session import get_user_from_session
from config.db import get_db

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router=APIRouter()

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

router = APIRouter()

class QuotaResponse(BaseModel):
    used: int
    limit: int
@router.get("/quota", response_model=QuotaResponse)
async def get_quota(session_id: str = Header(..., alias="X-Session-ID")):
    if not session_id:
        raise HTTPException(status_code=401, detail="Missing session")

    user = await get_user_from_session(session_id)
    db = await get_db()

    row = await db.fetchrow(
        """
        SELECT font_search_count,font_search_limit from users where id=$1
        """,
        user["id"]
    )

    if(not row):
        raise HTTPException(
        status_code=403,
        detail=(
            f"Font search limit reached. "
            f"{user['font_search_count']}/{user['font_search_limit']}"
        )
        )
    
    return {"used": row["font_search_count"], "limit": row["font_search_limit"]}


@router.post("/match-font")
async def match_font_endpoint(
        file: UploadFile = File(...),
        character: str = Form(...),
        top_k: int = Form(5),
        include_images: bool = Form(False),
        session_id: str = Header(..., alias="X-Session-ID")
    ):
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        if len(character) != 1:
            raise HTTPException(status_code=400, detail="Character must be a single character")

        if not 1 <= top_k <= 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

        try:
            user = await get_user_from_session(session_id)
            db = await get_db()

            # ───────── ATOMIC QUOTA CHECK + INCREMENT ─────────
            row = await db.fetchrow(
                """
                UPDATE users
                SET font_search_count = font_search_count + 1
                WHERE id = $1
                AND font_search_count < font_search_limit
                RETURNING font_search_count, font_search_limit
                """,
                user["id"]
            )

            if not row:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Font search limit reached. "
                        f"{user['font_search_count']}/{user['font_search_limit']}"
                    )
                )

            image_data = await file.read()
            if len(image_data) > 10 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

            matches = match_font(
                image_bytes=image_data,
                character=character,
                top_k=top_k,
                enhance=True,
                include_images=include_images
            )

            logger.info(
                f"Font search count: {row['font_search_count']}/{row['font_search_limit']}"
            )

            return {
                "character": character,
                "original_filename": file.filename,
                "matches": matches,
                "total_matches": len(matches),
                "message": f"Successfully matched {len(matches)} fonts"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/font-library/info")
async def get_font_library_info():
    try:
        info = get_library_info()
        return JSONResponse(content={
            "library": info,
            "status": "available",
            "message": "Font library is loaded and ready"
        })
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting library info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error accessing font library: {str(e)}"
        )


@router.get("/font-library/characters")
async def get_available_chars():
    try:
        chars = get_available_characters()
        return JSONResponse(content={
            "characters": chars,
            "count": len(chars),
            "message": f"{len(chars)} characters available for matching"
        })
    except Exception as e:
        logger.error(f"Error getting available characters: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error accessing font library: {str(e)}"
        )

