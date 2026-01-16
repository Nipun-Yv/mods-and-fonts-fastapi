import os
import torch
import pickle
import numpy as np
import ssl
import urllib3
from PIL import Image, ImageFont, ImageDraw
from torchvision import models, transforms
from tqdm import tqdm

# Disable SSL verification for model downloads (for development)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Configuration
FONT_DIR = "./my_fonts"         # Folder containing .ttf / .otf files
OUTPUT_FILE = "font_library.pkl"
ANCHOR_CHARS = "AagRS"          # Diverse structural characters
IMG_SIZE = 224

# 1. Load Feature Extractor (ResNet-18)
# We remove the final classification layer to get raw feature vectors
base_model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
model.eval()

# 2. Image Pre-processing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def render_char(font_path, char):
    """Renders a single character from a font file to a PIL Image."""
    try:
        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(0, 0, 0)) # Black background
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(IMG_SIZE * 0.7))
        
        # Center the character using textbbox (newer Pillow API)
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text(((IMG_SIZE-w)/2, (IMG_SIZE-h)/2), char, fill=(255, 255, 255), font=font)
        return img
    except Exception as e:
        print(f"Error rendering {char} from {font_path}: {e}")
        return None

def get_embedding(pil_img):
    """Converts a PIL image into a 512-dimensional vector."""
    tensor = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        vector = model(tensor)
    return vector.flatten().numpy()

# 3. Process the Library
# library = {}
# font_files = [f for f in os.listdir(FONT_DIR) if f.endswith(('.ttf', '.otf'))]

# for font_name in tqdm(font_files, desc="Embedding Fonts"):
#     font_path = os.path.join(FONT_DIR, font_name)
#     font_embeddings = {}
    
#     for char in ANCHOR_CHARS:
#         img = render_char(font_path, char)
#         if img:
#             font_embeddings[char] = get_embedding(img)
    
#     if font_embeddings:
#         library[font_name] = font_embeddings

# # Save to disk
# with open(OUTPUT_FILE, 'wb') as f:
#     pickle.dump(library, f)

# print(f"\nSuccess! Library of {len(library)} fonts saved to {OUTPUT_FILE}")


# ============== FONT MATCHING FUNCTIONALITY ==============

def load_font_library(library_path=OUTPUT_FILE):
    """Load the pre-built font library."""
    with open(library_path, 'rb') as f:
        return pickle.load(f)


def extract_character_from_image(image_path, char='A'):
    """
    Extract a character from an image for matching.
    This is a simplified version - assumes the image contains the character.
    """
    img = Image.open(image_path).convert('RGB')
    # Resize to match our training size
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return get_embedding(img)


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def find_matching_font(image_path, library_path=OUTPUT_FILE, char='A', top_k=5):
    """
    Find the best matching font for a character in an image.
    
    Args:
        image_path: Path to the image containing the character
        library_path: Path to the font library pickle file
        char: Which character to match against (default: 'A')
        top_k: Number of top matches to return
    
    Returns:
        List of tuples (font_name, similarity_score)
    """
    # Load library
    library = load_font_library(library_path)
    
    # Extract features from sample image
    sample_embedding = extract_character_from_image(image_path, char)
    
    # Compare against all fonts
    similarities = []
    for font_name, font_data in library.items():
        if char in font_data:
            similarity = cosine_similarity(sample_embedding, font_data[char])
            similarities.append((font_name, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def match_multiple_characters(image_path, library_path=OUTPUT_FILE, chars=None, top_k=5):
    """
    Match multiple characters and aggregate results.
    
    Args:
        image_path: Path to the image
        library_path: Path to the font library
        chars: List of characters to match (default: uses ANCHOR_CHARS)
        top_k: Number of top matches to return
    
    Returns:
        List of tuples (font_name, average_similarity_score)
    """
    if chars is None:
        chars = ANCHOR_CHARS
    
    library = load_font_library(library_path)
    font_scores = {}
    
    # For each character, get similarities
    for char in chars:
        sample_embedding = extract_character_from_image(image_path, char)
        
        for font_name, font_data in library.items():
            if char in font_data:
                similarity = cosine_similarity(sample_embedding, font_data[char])
                if font_name not in font_scores:
                    font_scores[font_name] = []
                font_scores[font_name].append(similarity)
    
    # Average scores for each font
    avg_scores = [(font, np.mean(scores)) for font, scores in font_scores.items()]
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    
    return avg_scores[:top_k]


if __name__ == "__main__":
    import sys
    
    # Check if we're building the library or matching
    if len(sys.argv) > 1 and sys.argv[1] == 'match':
        # Matching mode
        if not os.path.exists(OUTPUT_FILE):
            print(f"Error: Font library not found at {OUTPUT_FILE}")
            print("Run the script without arguments first to build the library.")
            sys.exit(1)
        
        image_path = sys.argv[2] if len(sys.argv) > 2 else "sample_image.png"
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"MATCHING FONT FOR: {image_path}")
        print(f"{'='*60}\n")
        
        # Try matching with character 'A'
        print("Matching single character 'A':")
        print("-" * 60)
        matches = find_matching_font(image_path, char='A', top_k=5)
        
        if matches:
            for i, (font_name, score) in enumerate(matches, 1):
                print(f"{i}. {font_name:40s} | Similarity: {score:.4f}")
        else:
            print("No matches found.")
        
        print("\n" + "="*60)
        print("Done!")
        
    else:
        # Building mode (existing code)
        print("Building font library...")
        print(f"Processing fonts from: {FONT_DIR}")
        print(f"Output will be saved to: {OUTPUT_FILE}")
        print("-" * 60)