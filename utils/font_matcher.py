"""
Font Matcher API Utilities using Vision Transformer (ViT-B/16)
Extracted from character_detection3.py for API use
"""

import os
import torch
import pickle
import numpy as np
import ssl
import urllib3
import base64
from PIL import Image, ImageEnhance
from io import BytesIO
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Disable SSL verification for model downloads
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_LIBRARY_PATH = os.path.join(SCRIPT_DIR, "font_library_vit.pkl")
FONT_FACES_DIR = os.path.join(SCRIPT_DIR, "font_faces")
IMG_SIZE = 224

# Global model instances
_vit_model = None
_vit_processor = None
_device = None
_font_library = None


def load_vit_model():
    """Load Vision Transformer model (lazy loading)."""
    global _vit_model, _vit_processor, _device
    
    if _vit_model is not None:
        return _vit_processor, _vit_model, _device
    
    try:
        from transformers import ViTModel, ViTImageProcessor
        
        logger.info("Loading Vision Transformer model...")
        _vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        _vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        _vit_model.eval()
        
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _vit_model = _vit_model.to(_device)
        
        logger.info(f"Vision Transformer loaded successfully on {_device}")
        return _vit_processor, _vit_model, _device
        
    except ImportError as e:
        logger.error("transformers library not available. Install with: pip install transformers")
        raise RuntimeError("Vision Transformer not available. Install transformers library.") from e
    except Exception as e:
        logger.error(f"Failed to load Vision Transformer: {str(e)}")
        raise


def load_font_library():
    """Load the pre-built font library (lazy loading)."""
    global _font_library
    
    if _font_library is not None:
        return _font_library
    
    if not os.path.exists(FONT_LIBRARY_PATH):
        raise FileNotFoundError(
            f"Font library not found at: {FONT_LIBRARY_PATH}\n"
            f"Please run: python utils/character_detection3.py build"
        )
    
    logger.info(f"Loading font library from {FONT_LIBRARY_PATH}...")
    with open(FONT_LIBRARY_PATH, 'rb') as f:
        _font_library = pickle.load(f)
    
    logger.info(f"Font library loaded with {len(_font_library)} fonts")
    return _font_library


def get_embedding(pil_img: Image.Image) -> np.ndarray:
    """
    Extract 768-dimensional features from an image using Vision Transformer.
    
    Args:
        pil_img: PIL Image to process
    
    Returns:
        768-dimensional numpy array
    """
    processor, model, device = load_vit_model()
    
    # Process image
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get features
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token embedding (first token)
        features = outputs.last_hidden_state[:, 0, :].cpu()
    
    return features.flatten().numpy()


def preprocess_image(image_bytes: bytes, enhance: bool = True) -> Image.Image:
    """
    Preprocess image from bytes for font matching.
    
    Args:
        image_bytes: Raw image bytes
        enhance: Apply contrast and sharpness enhancement
    
    Returns:
        Preprocessed PIL Image
    """
    # Load image from bytes
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    # Apply enhancement
    if enhance:
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Sharpness(img).enhance(1.5)
    
    # Resize to standard size
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    
    return img


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Returns:
        Similarity score between -1.0 and 1.0 (1.0 = identical)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def get_font_face_image(font_name: str, character: str) -> Optional[str]:
    """
    Get base64 encoded font face image.
    
    Args:
        font_name: Name of the font file
        character: Character in the font face image
    
    Returns:
        Base64 encoded PNG image or None if not found
    """
    # Construct expected filename
    base_name = os.path.splitext(font_name)[0]
    image_filename = f"{base_name}_{character}.png"
    image_path = os.path.join(FONT_FACES_DIR, image_filename)
    
    if os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to load font face image {image_filename}: {e}")
            return None
    
    return None


def match_font(
    image_bytes: bytes,
    character: str = 'A',
    top_k: int = 5,
    enhance: bool = True,
    include_images: bool = False
) -> List[Dict[str, any]]:
    """
    Find matching fonts for a character in an image.
    
    Args:
        image_bytes: Raw image bytes
        character: Character to match against (default: 'A')
        top_k: Number of top matches to return (default: 5)
        enhance: Apply image enhancement (default: True)
        include_images: Include base64 encoded font face images (default: False)
    
    Returns:
        List of dictionaries with:
        - font_name: Name of the font file
        - similarity: Similarity score (0.0 to 1.0)
        - rank: Rank in results (1-based)
        - image: Base64 encoded PNG (only if include_images=True)
    """
    # Load font library
    library = load_font_library()
    
    # Validate character
    if not character or len(character) != 1:
        raise ValueError("Character must be a single character")
    
    # Check if character exists in library
    fonts_with_char = sum(1 for font_data in library.values() if character in font_data)
    if fonts_with_char == 0:
        raise ValueError(f"Character '{character}' not found in font library. Try: A, a, B, b, E, e, G, g, M, m, R, r, S, s, T, t")
    
    # Preprocess image and extract features
    logger.info(f"Matching character '{character}' in image...")
    img = preprocess_image(image_bytes, enhance)
    sample_embedding = get_embedding(img)
    
    # Compare against all fonts
    matches = []
    for font_name, font_data in library.items():
        if character in font_data:
            similarity = cosine_similarity(sample_embedding, font_data[character])
            matches.append({
                'font_name': font_name,
                'similarity': similarity
            })
    
    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Add rank, font face images, and limit to top_k
    results = []
    for rank, match in enumerate(matches[:top_k], 1):
        result = {
            'font_name': match['font_name'],
            'similarity': round(match['similarity'], 4),
            'rank': rank
        }
        
        # Add font face image if requested
        if include_images:
            font_image = get_font_face_image(match['font_name'], character)
            if font_image:
                result['font_face_image'] = font_image
            else:
                logger.warning(f"Font face image not found for {match['font_name']}")
        
        results.append(result)
    
    logger.info(f"Found {len(results)} matching fonts for character '{character}'")
    return results


def get_available_characters() -> List[str]:
    """
    Get list of characters available in the font library.
    
    Returns:
        List of characters that can be matched
    """
    library = load_font_library()
    
    # Get all unique characters across all fonts
    chars = set()
    for font_data in library.values():
        chars.update(font_data.keys())
    
    return sorted(list(chars))


def get_library_info() -> Dict[str, any]:
    """
    Get information about the loaded font library.
    
    Returns:
        Dictionary with library statistics
    """
    library = load_font_library()
    
    total_fonts = len(library)
    available_chars = get_available_characters()
    
    # Calculate average characters per font
    total_chars = sum(len(font_data) for font_data in library.values())
    avg_chars_per_font = total_chars / total_fonts if total_fonts > 0 else 0
    
    return {
        'total_fonts': total_fonts,
        'available_characters': available_chars,
        'avg_chars_per_font': round(avg_chars_per_font, 1),
        'library_path': FONT_LIBRARY_PATH,
        'model': 'Vision Transformer (ViT-B/16)',
        'feature_dimension': 768
    }
