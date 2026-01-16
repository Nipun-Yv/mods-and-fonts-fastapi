from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import aiohttp
import base64
import os
from typing import List, Optional
import logging

from dotenv import load_dotenv
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware
   

env_path = Path(__file__).parent.parent / '.env'

load_dotenv(dotenv_path=env_path)

GEMINI_API_URL = os.getenv(
    "GEMINI_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-image:generateContent"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
async def generate_single_design(
    session: aiohttp.ClientSession,
    image_base64: str,
    mime_type: str,
    prompt: str,
    variation_number: int = 1,
    user_prompt: Optional[str] = None
) -> str:
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Build variation prompt with optional user prompt
    user_prompt_section = ""
    if user_prompt:
        user_prompt_section = f"\n\nUSER REQUEST: {user_prompt}\nIncorporate the user's request into this variation."
    
    variation_prompt = f"""{prompt}

Variation {variation_number}: Analyze the sketch overlay carefully and identify:
- What design elements are being added or modified (shapes, patterns, textures, colors)
- How the sketch suggests enhancing or transforming the underlying image
- The style and aesthetic direction indicated by the sketch{user_prompt_section}

Generate a refined design that:
1. Embeds the sketch's suggested modifications into the base image
2. Applies rich colors, textures, and details where the sketch indicates
3. Maintains visual coherence between the sketch-based additions and the original image
4. Creates a unique interpretation while staying true to the sketch's design intent
5. Incorporates the user's specific requirements if provided

Make this variation distinct from other variations while following the sketch's guidance."""
    
    payload = {
        "contents": [{
            "parts": [
                {"text": variation_prompt},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_base64
                    }
                }
            ]
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"]
        }
    }
    
    async with session.post(
        GEMINI_API_URL,
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=120),
        ssl=False
    ) as response:
        if response.status == 200:
            result = await response.json()
            
            candidates = result.get("candidates", [])
            if not candidates:
                raise HTTPException(
                    status_code=500,
                    detail="No candidates returned from Gemini API"
                )
            
            # Check each candidate
            for candidate in candidates:
                content = candidate.get("content", {})
                if not content:
                    continue
                    
                parts = content.get("parts", [])
                if not parts:
                    continue
                
                # Check each part for image data
                for i, part in enumerate(parts):
             
                    # Check for inline_data (image) - this is the expected structure
                    if "inlineData" in part:
                        inline_data = part["inlineData"]
                        
        
                        if "data" in inline_data:
                            return inline_data["data"]
                    
                    elif "data" in part:
                        return part["data"]
            
            raise HTTPException(
                status_code=500,
                detail="No image data found in Gemini API response. Check logs for response structure."
            )
        else:
            error_message = await response.text()
            logger.error(f"Gemini API error: {error_message}")
            raise HTTPException(
                status_code=response.status,
                detail=f"Error from Gemini API: {error_message}"
            )


async def generate_designs(
    image_base64: str, 
    mime_type: str, 
    prompt: str,
    user_prompt: Optional[str] = None
) -> List[str]:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable is not set"
        )
    
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            design_1 = await generate_single_design(
                session, image_base64, mime_type, prompt, 
                variation_number=1, user_prompt=user_prompt
            )
            design_2 = await generate_single_design(
                session, image_base64, mime_type, prompt, 
                variation_number=2, user_prompt=user_prompt
            )
            
            return [design_1, design_2]
                    
    except HTTPException:
        raise
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to Gemini API: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


async def generate_refined_design(
    image_base64: str, 
    mime_type: str, 
    base_prompt: str,
    refinement_instructions: str
) -> str:
    """
    Generate a refined design based on specific instructions.
    This is used for iterative refinement of designs.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable is not set"
        )
    
    refined_prompt = f"""{base_prompt}

REFINEMENT REQUEST:
{refinement_instructions}

Carefully analyze the image and apply the requested refinements. Focus on:
- Understanding what specific changes are being requested
- Maintaining the overall design coherence
- Enhancing the sketch-based modifications as specified
- Creating a polished, refined version that incorporates the instructions"""
    
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            refined_design = await generate_single_design(
                session, 
                image_base64, 
                mime_type, 
                refined_prompt, 
                variation_number=1
            )
            return refined_design
                    
    except HTTPException:
        raise
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to Gemini API: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
