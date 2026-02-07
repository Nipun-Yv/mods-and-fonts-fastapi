from fastapi import File, UploadFile, HTTPException, Form,APIRouter
from fastapi.responses import JSONResponse
from utils.font_matcher import match_font, get_available_characters, get_library_info
from typing import Optional

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router=APIRouter()


@router.post("/match-font")
async def match_font_endpoint(
    file: UploadFile = File(..., description="Image file containing a character"),
    character: str = Form(..., description="Single character to match (e.g., 'A', 'a', 'B')"),
    top_k: Optional[int] = Form(5, description="Number of top matches to return (default: 5)"),
    include_images: Optional[bool] = Form(False, description="Include font face preview images (default: False)")
):
    """
    Match a font based on a character in an image using Vision Transformer.
    
    This endpoint uses state-of-the-art Vision Transformer (ViT) model to analyze
    the character in the image and find the best matching fonts from the library.
    
    Returns the top matching fonts with similarity scores.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    if not character or len(character) != 1:
        raise HTTPException(
            status_code=400,
            detail="Character must be a single character (e.g., 'A', 'a', 'B')"
        )
    
    if top_k < 1 or top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20"
        )
    
    try:
        image_data = await file.read()
        max_size = 10 * 1024 * 1024 
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Image file too large. Maximum size is {max_size / (1024*1024)}MB"
            )
        
        matches = match_font(
            image_bytes=image_data,
            character=character,
            top_k=top_k,
            enhance=True,
            include_images=include_images
        )
        
        return JSONResponse(content={
            "character": character,
            "original_filename": file.filename,
            "matches": matches,
            "total_matches": len(matches),
            "model": "Vision Transformer (ViT-B/16)",
            "message": f"Successfully matched {len(matches)} fonts for character '{character}'"
        })
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Font matching service not available: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error matching font: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@router.get("/font-library/info")
async def get_font_library_info():
    """
    Get information about the font library.
    
    Returns statistics about available fonts, characters, and the model used.
    """
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
    """
    Get list of characters available for matching.
    
    Returns all characters that are available in the font library.
    """
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

