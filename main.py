from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from utils.gemini_design_gen import generate_designs, generate_refined_design
from utils.font_matcher import match_font, get_available_characters, get_library_info
import base64
import os
from typing import List, Optional
import logging
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

MASTER_PROMPT = os.getenv("MASTER_PROMPT", "You are an expert at interpreting sketch overlays and wireframes drawn on images. Your task is to: 1. Identify the sketch/drawing overlay elements, 2. Understand the design modifications and enhancements suggested by the sketch, 3. Apply these modifications to the underlying image with rich colors, textures, and details, 4. Seamlessly integrate the sketch-based design elements while preserving the original image context. The sketch represents design intent - bring it to life while maintaining visual harmony with the base image. Remeber to remove the sketch elements from the final image and replace them with real stuff. Respect the boundaries and position of rough brush strokes/sketches you see on the image.")

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




@app.post("/generate-designs")
async def generate_design_variations(
    file: UploadFile = File(..., description="Image file with sketch overlay to generate design variations from"),
    user_prompt: Optional[str] = Form(None, description="Optional user text prompt to guide the design generation"),
    refinement_instructions: Optional[str] = Form(None, description="Optional refinement instructions to enhance the design")
):
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
        
        # Enhance prompt with refinement instructions if provided
        enhanced_prompt = MASTER_PROMPT
        if refinement_instructions:
            enhanced_prompt = f"{MASTER_PROMPT}\n\nAdditional refinement instructions: {refinement_instructions}"
        
        design_variations = await generate_designs(
            encoded_image, 
            file.content_type, 
            enhanced_prompt,
            user_prompt=user_prompt
        )
        
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



@app.get("/")
async def root():
    return {
        "message": "Image Design Generator API",
        "status": "running",
        "endpoints": {
            "generate_designs": "/generate-designs (POST) - Generate design variations from sketch overlay",
            "refine_design": "/refine-design (POST) - Refine a design with specific instructions",
            "segment_image": "/segment-image (POST) - Segment image into objects",
            "match_font": "/match-font (POST) - Match font using Vision Transformer AI",
            "font_library_info": "/font-library/info (GET) - Get font library statistics",
            "font_library_characters": "/font-library/characters (GET) - Get available characters"
        }
    }


@app.post("/refine-design")
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


@app.post("/match-font")
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
    
    # Validate character
    if not character or len(character) != 1:
        raise HTTPException(
            status_code=400,
            detail="Character must be a single character (e.g., 'A', 'a', 'B')"
        )
    
    # Validate top_k
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
        
        # Match font using Vision Transformer
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


@app.get("/font-library/info")
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


@app.get("/font-library/characters")
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


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
