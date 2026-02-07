from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Header
from fastapi.responses import JSONResponse
from utils.gemini_design_gen import generate_designs, generate_refined_design
from config.db import get_db
import base64
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
MASTER_PROMPT = os.getenv("MASTER_PROMPT", "You are an expert at interpreting sketch overlays and wireframes drawn on images. Your task is to: 1. Identify the sketch/drawing overlay elements, 2. Understand the design modifications and enhancements suggested by the sketch, 3. Apply these modifications to the underlying image with rich colors, textures, and details, 4. Seamlessly integrate the sketch-based design elements while preserving the original image context. The sketch represents design intent - bring it to life while maintaining visual harmony with the base image. Remeber to remove the sketch elements from the final image and replace them with real stuff. Respect the boundaries and position of rough brush strokes/sketches you see on the image.")


router=APIRouter()

async def get_user_from_token(session_id: str):
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

@router.post("/generate-designs")
async def generate_design_variations(
    file: UploadFile = File(..., description="Image file with sketch overlay to generate design variations from"),
    user_prompt: Optional[str] = Form(None, description="Optional user text prompt to guide the design generation"),
    refinement_instructions: Optional[str] = Form(None, description="Optional refinement instructions to enhance the design"),
    session_id: str = Header(..., alias="X-Session-ID")
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        user = await get_user_from_token(session_id)
        if user['generation_count'] >= user['generation_limit']:
            raise HTTPException(
                status_code=403,
                detail=f"Generation limit reached. {user['generation_count']}/{user['generation_limit']}"
            )
        
        image_data = await file.read()
        max_size = 10 * 1024 * 1024 
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Image file too large. Maximum size is {max_size / (1024*1024)}MB"
            )
        
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        

        enhanced_prompt = MASTER_PROMPT
        if refinement_instructions:
            enhanced_prompt = f"{MASTER_PROMPT}\n\nAdditional refinement instructions: {refinement_instructions}"
        
        design_variations = await generate_designs(
            encoded_image, 
            file.content_type, 
            enhanced_prompt,
            user_prompt=user_prompt
        )
        db = await get_db()
        await db.execute(
            'UPDATE users SET generation_count = generation_count + 1 WHERE id = $1',
            user['id']
        )
        
        new_count = user['generation_count'] + 1
        logger.info(f"✅ Generation count: {new_count}/{user['generation_limit']}")

        return JSONResponse(content={
            "original_image": encoded_image,
            "original_filename": file.filename,
            "design_variations": design_variations,
            "user_prompt": user_prompt,
            "refinement_applied": refinement_instructions is not None,
            "message": "Successfully generated 2 design variations"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )



@router.post("/refine-design")
async def refine_design_endpoint(
    file: UploadFile = File(..., description="Image file with sketch overlay to refine"),
    refinement_instructions: str = Form(..., description="Specific instructions for refining the design"),
    base_variation: Optional[str] = Form(None, description="Base64 encoded variation to refine (optional)")
):
    """
    Refine a design based on specific instructions.
    Takes an image with sketch overlay and refinement instructions to generate an improved version.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        image_data = await file.read()
        max_size = 10 * 1024 * 1024 
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Image file too large. Maximum size is {max_size / (1024*1024)}MB"
            )
        
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        # Use base variation if provided, otherwise use original
        base_image = base_variation if base_variation else encoded_image
        
        refined_design = await generate_refined_design(
            base_image, 
            file.content_type, 
            MASTER_PROMPT,
            refinement_instructions
        )
        
        return JSONResponse(content={
            "original_image": encoded_image,
            "refined_design": refined_design,
            "refinement_instructions": refinement_instructions,
            "message": "Successfully generated refined design"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refining design: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

