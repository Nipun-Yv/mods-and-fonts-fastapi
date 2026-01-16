"""
Enhanced Font Matching using DenseNet-169
A denser, more powerful model for better character/font recognition.
DenseNet has better gradient flow and feature reuse compared to ResNet.
"""

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
FONT_DIR = "./my_fonts"              # Folder containing .ttf / .otf files
OUTPUT_FILE = "font_library_dense.pkl"
ANCHOR_CHARS = "AaBbGgRrSs"         # More anchor characters for better matching
IMG_SIZE = 224

print("Loading DenseNet-169 (denser, more powerful model)...")
print("This model has 169 layers with dense connections for better feature extraction")

# 1. Load Feature Extractor (DenseNet-169)
# DenseNet connects each layer to every other layer in a feed-forward fashion
# This creates denser feature maps and better gradient flow
base_model = models.densenet169(pretrained=True)

# Remove the final classification layer to get feature vectors
# DenseNet outputs 1664-dimensional features (vs ResNet's 512)
model = torch.nn.Sequential(
    base_model.features,
    torch.nn.AdaptiveAvgPool2d((1, 1)),
    torch.nn.Flatten()
)
model.eval()

print(f"Model loaded! Feature dimension: 1664 (vs ResNet's 512)")
print(f"Using {len(ANCHOR_CHARS)} anchor characters for robust matching\n")

# 2. Image Pre-processing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def render_char(font_path, char, size=IMG_SIZE):
    """Renders a single character from a font file to a PIL Image."""
    try:
        img = Image.new('RGB', (size, size), color=(0, 0, 0))  # Black background
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(size * 0.7))
        
        # Center the character using textbbox
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (size - w) / 2
        y = (size - h) / 2
        draw.text((x, y), char, fill=(255, 255, 255), font=font)
        return img
    except Exception as e:
        print(f"  ⚠️  Error rendering '{char}' from {os.path.basename(font_path)}: {e}")
        return None


def get_embedding(pil_img):
    """
    Converts a PIL image into a 1664-dimensional vector using DenseNet-169.
    Higher dimensional features provide better discrimination.
    """
    tensor = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        vector = model(tensor)
    return vector.flatten().numpy()


def build_font_library(font_dir=FONT_DIR, output_file=OUTPUT_FILE):
    """Build a font library with DenseNet embeddings."""
    library = {}
    font_files = [f for f in os.listdir(font_dir) if f.endswith(('.ttf', '.otf'))]
    
    if not font_files:
        print(f"❌ No font files found in {font_dir}")
        return None
    
    print(f"Found {len(font_files)} font files")
    print(f"Processing {len(ANCHOR_CHARS)} characters per font...")
    print("-" * 60)
    
    for font_name in tqdm(font_files, desc="📚 Building Font Library"):
        font_path = os.path.join(font_dir, font_name)
        font_embeddings = {}
        
        for char in ANCHOR_CHARS:
            img = render_char(font_path, char)
            if img:
                font_embeddings[char] = get_embedding(img)
        
        # Only add if we successfully rendered at least half the characters
        if len(font_embeddings) >= len(ANCHOR_CHARS) // 2:
            library[font_name] = font_embeddings
    
    # Save to disk
    with open(output_file, 'wb') as f:
        pickle.dump(library, f)
    
    print(f"\n✅ Success! Library of {len(library)} fonts saved to {output_file}")
    print(f"📊 Average features per font: {sum(len(v) for v in library.values()) / len(library):.1f} characters")
    return library


# ============== FONT MATCHING FUNCTIONALITY ==============

def load_font_library(library_path=OUTPUT_FILE):
    """Load the pre-built font library."""
    if not os.path.exists(library_path):
        raise FileNotFoundError(f"Font library not found: {library_path}\nRun with 'build' mode first.")
    
    with open(library_path, 'rb') as f:
        return pickle.load(f)


def extract_character_from_image(image_path, preprocess_image=True):
    """
    Extract character features from an image.
    
    Args:
        image_path: Path to the image
        preprocess_image: If True, apply preprocessing (recommended)
    """
    img = Image.open(image_path).convert('RGB')
    
    if preprocess_image:
        # Convert to grayscale and back to RGB for better feature extraction
        import numpy as np
        img_array = np.array(img)
        
        # Simple preprocessing: normalize
        if img_array.max() > 1:
            img_array = img_array / 255.0
        
        img = Image.fromarray((img_array * 255).astype(np.uint8))
    
    # Resize to match training size
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return get_embedding(img)


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1, vec2):
    """Calculate normalized Euclidean distance (0=identical, 1=very different)."""
    dist = np.linalg.norm(vec1 - vec2)
    # Normalize to 0-1 range (approximate)
    max_dist = np.sqrt(len(vec1))  # Maximum possible distance
    return min(dist / max_dist, 1.0)


def find_matching_font(image_path, library_path=OUTPUT_FILE, char='A', top_k=5, 
                       metric='cosine', preprocess=True):
    """
    Find the best matching font for a character in an image.
    
    Args:
        image_path: Path to the image containing the character
        library_path: Path to the font library pickle file
        char: Which character to match against (default: 'A')
        top_k: Number of top matches to return
        metric: 'cosine' or 'euclidean' similarity metric
        preprocess: Whether to preprocess the image
    
    Returns:
        List of tuples (font_name, similarity_score)
    """
    library = load_font_library(library_path)
    sample_embedding = extract_character_from_image(image_path, preprocess)
    
    similarities = []
    for font_name, font_data in library.items():
        if char in font_data:
            if metric == 'cosine':
                score = cosine_similarity(sample_embedding, font_data[char])
            else:  # euclidean
                score = 1.0 - euclidean_distance(sample_embedding, font_data[char])
            similarities.append((font_name, score))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def match_multiple_characters(image_path, library_path=OUTPUT_FILE, chars=None, 
                              top_k=5, metric='cosine', preprocess=True):
    """
    Match multiple characters and aggregate results for more robust matching.
    
    Args:
        image_path: Path to the image
        library_path: Path to the font library
        chars: List of characters to match (default: uses ANCHOR_CHARS)
        top_k: Number of top matches to return
        metric: Similarity metric ('cosine' or 'euclidean')
        preprocess: Whether to preprocess the image
    
    Returns:
        List of tuples (font_name, average_similarity_score, matched_chars_count)
    """
    if chars is None:
        chars = ANCHOR_CHARS
    
    library = load_font_library(library_path)
    font_scores = {}
    
    for char in chars:
        try:
            sample_embedding = extract_character_from_image(image_path, preprocess)
            
            for font_name, font_data in library.items():
                if char in font_data:
                    if metric == 'cosine':
                        score = cosine_similarity(sample_embedding, font_data[char])
                    else:
                        score = 1.0 - euclidean_distance(sample_embedding, font_data[char])
                    
                    if font_name not in font_scores:
                        font_scores[font_name] = []
                    font_scores[font_name].append(score)
        except Exception as e:
            print(f"⚠️  Skipped character '{char}': {e}")
            continue
    
    # Calculate average scores and count
    results = []
    for font, scores in font_scores.items():
        avg_score = np.mean(scores)
        results.append((font, avg_score, len(scores)))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def visualize_top_matches(image_path, matches, output_dir="./matched_fonts"):
    """
    Render the top matching fonts for comparison.
    
    Args:
        image_path: Original image path
        matches: List of (font_name, score) tuples
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (font_name, score) in enumerate(matches, 1):
        font_path = os.path.join(FONT_DIR, font_name)
        if os.path.exists(font_path):
            # Render sample characters
            img = Image.new('RGB', (IMG_SIZE * len(ANCHOR_CHARS), IMG_SIZE), color=(255, 255, 255))
            
            for j, char in enumerate(ANCHOR_CHARS):
                char_img = render_char(font_path, char)
                if char_img:
                    img.paste(char_img, (j * IMG_SIZE, 0))
            
            output_path = os.path.join(output_dir, f"{i}_{font_name}_{score:.4f}.png")
            img.save(output_path)
    
    print(f"\n📁 Visualizations saved to: {output_dir}")


# ============== MAIN EXECUTION ==============

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'build':
            # Build the font library
            print("=" * 60)
            print("BUILDING FONT LIBRARY WITH DENSENET-169")
            print("=" * 60 + "\n")
            build_font_library()
            
        elif mode == 'match':
            # Match mode
            image_path = sys.argv[2] if len(sys.argv) > 2 else "sample_image.png"
            
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                sys.exit(1)
            
            print("\n" + "=" * 60)
            print(f"🔍 MATCHING FONT FOR: {image_path}")
            print("=" * 60 + "\n")
            
            # Single character matching
            print("📝 Single Character Matching (char 'A'):")
            print("-" * 60)
            matches = find_matching_font(image_path, char='A', top_k=5, metric='cosine')
            
            if matches:
                for i, (font_name, score) in enumerate(matches, 1):
                    bar = "█" * int(score * 20)
                    print(f"{i}. {font_name:40s} | {bar:20s} {score:.4f}")
            else:
                print("No matches found.")
            
            print("\n" + "=" * 60)
            print("✅ Done!")
        
        else:
            print("Usage:")
            print("  Build library: python character_detection2.py build")
            print("  Match image:   python character_detection2.py match sample_image.png")
    else:
        print("=" * 60)
        print("DenseNet-169 Font Matcher")
        print("=" * 60)
        print("\nUsage:")
        print("  1. Build library:  python character_detection2.py build")
        print("  2. Match font:     python character_detection2.py match <image_path>")
        print("\nExample:")
        print("  python character_detection2.py match sample_image.png")
