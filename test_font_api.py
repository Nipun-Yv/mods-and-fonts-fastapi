"""
Test script for Font Matching API
"""

import requests
import json

# API base URL (adjust if needed)
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_font_library_info():
    """Test font library info endpoint"""
    print("=" * 60)
    print("Testing Font Library Info")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/font-library/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_available_characters():
    """Test available characters endpoint"""
    print("=" * 60)
    print("Testing Available Characters")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/font-library/characters")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total characters: {data['count']}")
    print(f"Characters: {', '.join(data['characters'])}\n")


def test_font_matching(image_path="utils/sample_image.png", character="A", with_images=False):
    """Test font matching endpoint"""
    print("=" * 60)
    print(f"Testing Font Matching - Character: '{character}'")
    if with_images:
        print("(With font face images)")
    print("=" * 60)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f, 'image/png')}
            data = {
                'character': character,
                'top_k': 5,
                'include_images': str(with_images).lower()
            }
            
            response = requests.post(f"{BASE_URL}/match-font", files=files, data=data)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ {result['message']}\n")
                print(f"Character: {result['character']}")
                print(f"Model: {result['model']}")
                print(f"\nTop {result['total_matches']} Matches:")
                print("-" * 60)
                
                for match in result['matches']:
                    bar_len = int(match['similarity'] * 30)
                    bar = "█" * bar_len + "░" * (30 - bar_len)
                    has_image = "🖼️ " if 'font_face_image' in match else ""
                    print(f"{match['rank']}. {has_image}{match['font_name']:35s} │{bar}│ {match['similarity']:.4f}")
                
                if with_images and result['matches']:
                    has_images = sum(1 for m in result['matches'] if 'font_face_image' in m)
                    print(f"\n📊 Font face images included: {has_images}/{len(result['matches'])}")
                
                print()
            else:
                print(f"Error: {response.json()}\n")
                
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        print("Please provide a valid image path\n")
    except Exception as e:
        print(f"❌ Error: {str(e)}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 FONT MATCHING API TEST SUITE")
    print("=" * 60 + "\n")
    
    # Run tests
    test_health()
    test_font_library_info()
    test_available_characters()
    test_font_matching(with_images=False)
    test_font_matching(with_images=True)
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60 + "\n")
