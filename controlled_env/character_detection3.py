"""
State-of-the-Art Font Matching using Vision Transformer (ViT-B/16)

Vision Transformers use self-attention mechanisms to capture fine-grained details
and spatial relationships in character shapes - perfect for font matching!

Why ViT is better for fonts:
- Attention mechanisms focus on stroke details, curves, serifs
- Better at capturing subtle typographic differences
- No inductive bias of CNNs - learns patterns from scratch
- State-of-the-art performance on fine-grained visual tasks
"""

import os
import torch
import pickle
import numpy as np
import ssl
import urllib3
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from tqdm import tqdm

# Disable SSL verification for model downloads
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Configuration
FONT_DIR = "./ttfs_express_only"
OUTPUT_FILE = "font_library_vit.pkl"
ANCHOR_CHARS = "AaBbEeGgMmRrSsTt"  # Even more diverse characters
IMG_SIZE = 224  # ViT standard input size

print("=" * 70)
print("🚀 VISION TRANSFORMER (ViT-B/16) - STATE-OF-THE-ART FONT MATCHER")
print("=" * 70)
print("\n✨ Why ViT is superior for font matching:")
print("  • Attention mechanisms capture stroke details and curves")
print("  • Better at subtle typographic differences (serifs, weights)")
print("  • 768-dimensional rich features (vs DenseNet's 1664)")
print("  • State-of-the-art on fine-grained recognition")
print("\n📊 Using {} anchor characters for robust matching".format(len(ANCHOR_CHARS)))
print("-" * 70)

try:
    from transformers import ViTModel, ViTImageProcessor
    VIT_AVAILABLE = True
    print("✅ Vision Transformer available!\n")
except ImportError:
    VIT_AVAILABLE = False
    print("❌ Vision Transformer not available.")
    print("Install with: pip install transformers")
    import sys
    sys.exit(1)

# Load Vision Transformer
print("📥 Loading ViT-B/16 (this may take a moment on first run)...")
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
model.eval()
print("✅ Model loaded! Feature dimension: 768\n")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"🖥️  Using device: {device}\n")


def render_char(font_path, char, size=IMG_SIZE, apply_antialiasing=True):
    """
    Renders a single character with high quality.
    
    Args:
        font_path: Path to font file
        char: Character to render
        size: Image size
        apply_antialiasing: Better quality rendering
    """
    try:
        # Use white background for better ViT performance (trained on natural images)
        img = Image.new('RGB', (size, size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(size * 0.65))
        
        # Center the character
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (size - w) / 2 - bbox[0]
        y = (size - h) / 2 - bbox[1]
        
        # Draw with black text on white background
        draw.text((x, y), char, fill=(0, 0, 0), font=font)
        
        return img
    except Exception as e:
        return None


def get_embedding(pil_img):
    """
    Extract 768-dimensional features using Vision Transformer.
    
    ViT processes the image as patches and uses multi-head self-attention
    to capture relationships between different parts of the character.
    """
    # Process image
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get features
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token embedding (first token)
        features = outputs.last_hidden_state[:, 0, :].cpu()
    
    return features.flatten().numpy()


def build_font_library(font_dir=FONT_DIR, output_file=OUTPUT_FILE):
    """Build font library with Vision Transformer embeddings."""
    library = {}
    font_files = [f for f in os.listdir(font_dir) if f.endswith(('.ttf', '.otf'))]
    
    if not font_files:
        print(f"❌ No font files found in {font_dir}")
        return None
    
    print("=" * 70)
    print(f"📚 BUILDING FONT LIBRARY")
    print("=" * 70)
    print(f"  Fonts found: {len(font_files)}")
    print(f"  Characters per font: {len(ANCHOR_CHARS)}")
    print(f"  Total embeddings: {len(font_files) * len(ANCHOR_CHARS)}")
    print("-" * 70 + "\n")
    
    successful = 0
    failed = 0
    
    for font_name in tqdm(font_files, desc="🎨 Embedding Fonts", ncols=70):
        font_path = os.path.join(font_dir, font_name)
        font_embeddings = {}
        char_count = 0
        
        for char in ANCHOR_CHARS:
            img = render_char(font_path, char)
            if img:
                try:
                    font_embeddings[char] = get_embedding(img)
                    char_count += 1
                except Exception as e:
                    continue
        
        # Require at least 70% of characters to be successfully rendered
        if char_count >= len(ANCHOR_CHARS) * 0.7:
            library[font_name] = font_embeddings
            successful += 1
        else:
            failed += 1
    
    # Save to disk
    with open(output_file, 'wb') as f:
        pickle.dump(library, f)
    
    print(f"\n{'=' * 70}")
    print(f"✅ SUCCESS!")
    print(f"{'=' * 70}")
    print(f"  📦 Fonts processed: {successful} successful, {failed} failed")
    print(f"  💾 Saved to: {output_file}")
    print(f"  📊 Avg chars/font: {sum(len(v) for v in library.values()) / len(library):.1f}")
    print(f"  🎯 Library ready for matching!\n")
    
    return library


# ============== ADVANCED MATCHING FUNCTIONS ==============

def load_font_library(library_path=OUTPUT_FILE):
    """Load the pre-built font library."""
    if not os.path.exists(library_path):
        raise FileNotFoundError(
            f"❌ Font library not found: {library_path}\n"
            f"   Run: python character_detection3.py build"
        )
    
    with open(library_path, 'rb') as f:
        library = pickle.load(f)
    
    print(f"✅ Loaded library with {len(library)} fonts")
    return library


def extract_from_image(image_path, enhance=True):
    """
    Extract features from an image using ViT.
    
    Args:
        image_path: Path to image
        enhance: Apply image enhancement
    """
    img = Image.open(image_path).convert('RGB')
    
    if enhance:
        from PIL import ImageEnhance
        # Enhance contrast and sharpness
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Sharpness(img).enhance(1.5)
    
    # Resize to standard size
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    
    return get_embedding(img)


def cosine_similarity(vec1, vec2):
    """Cosine similarity: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def find_matching_font(image_path, library_path=OUTPUT_FILE, char='C', 
                       top_k=10, enhance=True):
    """
    Find matching fonts using Vision Transformer features.
    
    Args:
        image_path: Path to image containing character
        library_path: Path to font library
        char: Character to match
        top_k: Number of top matches
        enhance: Apply image enhancement
    
    Returns:
        List of (font_name, similarity_score) tuples
    """
    library = load_font_library(library_path)
    sample_embedding = extract_from_image(image_path, enhance)
    
    matches = []
    for font_name, font_data in library.items():
        if char in font_data:
            similarity = cosine_similarity(sample_embedding, font_data[char])
            matches.append((font_name, similarity))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]


def match_with_ensemble(image_path, library_path=OUTPUT_FILE, chars=None, 
                       top_k=10, enhance=True, weighting='uniform'):
    """
    Advanced ensemble matching with multiple characters.
    
    Uses multiple characters and combines their scores for robust matching.
    
    Args:
        image_path: Path to image
        library_path: Font library path
        chars: Characters to use (default: ANCHOR_CHARS)
        top_k: Number of results
        enhance: Image enhancement
        weighting: 'uniform' or 'confidence' based weighting
    
    Returns:
        List of (font_name, score, confidence, matched_chars)
    """
    if chars is None:
        chars = list(ANCHOR_CHARS[:5])  # Use first 5 by default
    
    library = load_font_library(library_path)
    font_scores = {}
    
    print(f"\n🔍 Analyzing with {len(chars)} characters: {chars}")
    print("-" * 70)
    
    for char in chars:
        try:
            sample_embedding = extract_from_image(image_path, enhance)
            
            for font_name, font_data in library.items():
                if char in font_data:
                    similarity = cosine_similarity(sample_embedding, font_data[char])
                    
                    if font_name not in font_scores:
                        font_scores[font_name] = []
                    font_scores[font_name].append(similarity)
        except Exception as e:
            print(f"  ⚠️  Skipped '{char}': {e}")
            continue
    
    # Calculate ensemble scores
    results = []
    for font, scores in font_scores.items():
        if weighting == 'confidence':
            # Weight by confidence (higher scores get more weight)
            weights = np.array(scores)
            avg_score = np.average(scores, weights=weights)
        else:
            avg_score = np.mean(scores)
        
        confidence = np.std(scores)  # Lower std = more consistent = higher confidence
        results.append((font, avg_score, 1.0 - min(confidence, 1.0), len(scores)))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def visualize_matches(matches, output_dir="./vit_matched_fonts", show_chars=None):
    """
    Create visual comparison of matched fonts.
    
    Args:
        matches: List of match results
        output_dir: Output directory
        show_chars: Characters to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if show_chars is None:
        show_chars = ANCHOR_CHARS[:8]
    
    print(f"\n🎨 Generating visualizations...")
    
    for i, match_data in enumerate(matches[:5], 1):
        font_name = match_data[0]
        font_path = os.path.join(FONT_DIR, font_name)
        
        if os.path.exists(font_path):
            # Create visualization
            char_w = IMG_SIZE
            img = Image.new('RGB', (char_w * len(show_chars), IMG_SIZE), (255, 255, 255))
            
            for j, char in enumerate(show_chars):
                char_img = render_char(font_path, char)
                if char_img:
                    img.paste(char_img, (j * char_w, 0))
            
            score = match_data[1]
            output_path = os.path.join(output_dir, f"{i:02d}_{font_name.replace('.ttf', '')}_{score:.4f}.png")
            img.save(output_path)
    
    print(f"✅ Saved to: {output_dir}")


# ============== MAIN EXECUTION ==============

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'build':
            build_font_library()
            
        elif mode == 'match':
            image_path = sys.argv[2] if len(sys.argv) > 2 else "sample_image.png"
            
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                sys.exit(1)
            
            print("\n" + "=" * 70)
            print(f"🎯 VISION TRANSFORMER FONT MATCHING")
            print("=" * 70)
            print(f"📸 Image: {image_path}\n")
            
            # Single character matching
            print("📊 Single Character Match (char 'A'):")
            print("-" * 70)
            matches = find_matching_font(image_path, char='C', top_k=10)
            
            if matches:
                for i, (font_name, score) in enumerate(matches, 1):
                    # Visual bar
                    bar_len = int(score * 30)
                    bar = "█" * bar_len + "░" * (30 - bar_len)
                    confidence = "🔥" if score > 0.85 else "✨" if score > 0.75 else "💫"
                    
                    print(f"{i:2d}. {confidence} {font_name:35s} │{bar}│ {score:.4f}")
            else:
                print("❌ No matches found.")
            
            # Ensemble matching
            print("\n" + "=" * 70)
            print("🎯 Ensemble Match (Multiple Characters):")
            print("-" * 70)
            ensemble = match_with_ensemble(image_path, top_k=5)
            
            if ensemble:
                for i, (font, score, conf, count) in enumerate(ensemble, 1):
                    bar_len = int(score * 30)
                    bar = "█" * bar_len + "░" * (30 - bar_len)
                    confidence_icon = "🎯" if conf > 0.8 else "🎲"
                    
                    print(f"{i}. {confidence_icon} {font:35s} │{bar}│ {score:.4f} (conf: {conf:.2f})")
            
            # Visualize top matches
            visualize_matches(matches[:5] if matches else [])
            
            print("\n" + "=" * 70)
            print("✅ MATCHING COMPLETE!")
            print("=" * 70)
        
        else:
            print("Usage:")
            print("  Build: python character_detection3.py build")
            print("  Match: python character_detection3.py match <image>")
    else:
        print("\n" + "=" * 70)
        print("Vision Transformer Font Matcher")
        print("=" * 70)
        print("\n📖 Usage:")
        print("  1. Build library:  python character_detection3.py build")
        print("  2. Match font:     python character_detection3.py match sample_image.png")
        print("\n🚀 Features:")
        print("  • State-of-the-art attention-based matching")
        print("  • 768-dim rich feature extraction")
        print("  • Ensemble matching for robustness")
        print("  • Visual similarity bars and confidence scores")
        print("")
