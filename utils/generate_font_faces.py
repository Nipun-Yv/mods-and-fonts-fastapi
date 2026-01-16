"""
Generate Font Face Images
Creates preview images for each font in the library showing character 'A'
"""

import os
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from tqdm import tqdm

# Configuration
FONT_DIR = "./my_fonts"
OUTPUT_DIR = "./font_faces"
CHARACTER = "A"
IMG_SIZE = 224

def render_font_face(font_path, char=CHARACTER, size=IMG_SIZE, output_path=None):
    """
    Render a character from a font file.
    
    Args:
        font_path: Path to font file
        char: Character to render
        size: Image size
        output_path: Optional path to save the image
    
    Returns:
        PIL Image
    """
    try:
        # White background for consistency with ViT training
        img = Image.new('RGB', (size, size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(size * 0.65))
        
        # Center the character
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (size - w) / 2 - bbox[0]
        y = (size - h) / 2 - bbox[1]
        
        # Draw with black text
        draw.text((x, y), char, fill=(0, 0, 0), font=font)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
        
        return img
    except Exception as e:
        print(f"  ⚠️  Error rendering '{char}' from {os.path.basename(font_path)}: {e}")
        return None


def generate_all_font_faces(font_dir=FONT_DIR, output_dir=OUTPUT_DIR, char=CHARACTER):
    """
    Generate font face images for all fonts.
    
    Args:
        font_dir: Directory containing font files
        output_dir: Directory to save generated images
        char: Character to render
    
    Returns:
        Dictionary mapping font names to output paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all font files
    font_files = [f for f in os.listdir(font_dir) if f.endswith(('.ttf', '.otf'))]
    
    if not font_files:
        print(f"❌ No font files found in {font_dir}")
        return {}
    
    print("=" * 70)
    print(f"🎨 GENERATING FONT FACE IMAGES")
    print("=" * 70)
    print(f"  Character: '{char}'")
    print(f"  Fonts: {len(font_files)}")
    print(f"  Output: {output_dir}")
    print("-" * 70 + "\n")
    
    generated = {}
    successful = 0
    failed = 0
    
    for font_name in tqdm(font_files, desc="Generating Faces", ncols=70):
        font_path = os.path.join(font_dir, font_name)
        
        # Create output filename (remove extension, add char and .png)
        base_name = os.path.splitext(font_name)[0]
        output_filename = f"{base_name}_{char}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Generate image
        img = render_font_face(font_path, char, IMG_SIZE, output_path)
        
        if img:
            generated[font_name] = output_path
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'=' * 70}")
    print(f"✅ GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print()
    
    return generated


if __name__ == "__main__":
    import sys
    
    # Allow custom character from command line
    char = sys.argv[1] if len(sys.argv) > 1 else CHARACTER
    
    if len(char) != 1:
        print("❌ Please provide a single character")
        print("Usage: python generate_font_faces.py [character]")
        print("Example: python generate_font_faces.py A")
        sys.exit(1)
    
    print(f"\n🎯 Generating font faces for character: '{char}'\n")
    generated = generate_all_font_faces(char=char)
    
    print(f"Generated {len(generated)} font face images")
    print(f"Files saved in: {OUTPUT_DIR}")
