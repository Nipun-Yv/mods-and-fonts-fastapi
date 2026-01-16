from json import load
import numpy as np
import cv2
import torch
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
import urllib.request
import ssl
import os

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

_sam2_model = None
_mask_generator = None
_device = None

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    logger.warning("SAM2 not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")


def load_sam_model():
    global _sam2_model, _mask_generator, _device
    
    if _sam2_model is not None:
        return True
    
    if not SAM2_AVAILABLE:
        logger.error("SAM2 is not installed")
        return False
    
    try:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {_device}")
        
        checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        checkpoint_path = PROJECT_ROOT / "models" / "sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        
        # Download checkpoint if it doesn't exist
        if not checkpoint_path.exists():
            logger.info(f"Downloading SAM2 checkpoint from {checkpoint_url}...")
            os.makedirs(checkpoint_path.parent, exist_ok=True)
            
            # Create SSL context that doesn't verify certificates (for development)
            # In production, you should use proper certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Download with SSL context
            req = urllib.request.Request(checkpoint_url)
            with urllib.request.urlopen(req, context=ssl_context) as response:
                with open(checkpoint_path, 'wb') as out_file:
                    out_file.write(response.read())
            
            logger.info("Checkpoint downloaded successfully")
        
        logger.info("Loading SAM2 model...")
        _sam2_model = build_sam2(
            model_cfg, 
            str(checkpoint_path), 
            device=_device, 
            apply_postprocessing=False
        )
        
        _mask_generator = SAM2AutomaticMaskGenerator(_sam2_model,
            points_per_side=32,
            points_per_batch=64,
            
            pred_iou_thresh=0.65,
            box_nms_thresh=0.4,
            stability_score_thresh=0.85,

            
            crop_n_layers=0,
            crop_n_points_downscale_factor=2, 
            min_mask_region_area=50,
            crop_overlap_ratio=0.2, 
            
            use_m2m=False,)
        
        logger.info("SAM2 model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load SAM2 model: {str(e)}")
        return False


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode a numpy image array (RGBA) to base64 PNG string."""
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    image_bytes = buffer.tobytes()
    return base64.b64encode(image_bytes).decode('utf-8')


def _extract_segments(
    image_bytes: bytes,
    include_crops: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Tuple[int, int]]:
    global _mask_generator

    if _mask_generator is None:
        raise RuntimeError("SAM2 model not loaded. Call load_sam_model() first.")

    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image from bytes")

    # Convert BGR to RGB for SAM2
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Generate masks
    logger.info(f"Generating masks for image of size {w}x{h}...")
    masks = _mask_generator.generate(image_rgb)
    logger.info(f"Found {len(masks)} segments")

    # Prepare mask records with bounding boxes
    mask_records = []
    for idx, mask_data in enumerate(masks):
        mask = mask_data["segmentation"].astype(np.uint8)
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            continue

        y_min, y_max = int(y_indices.min()), int(y_indices.max())
        x_min, x_max = int(x_indices.min()), int(x_indices.max())
        area = int(mask.sum())

        mask_records.append({
            "id": idx,
            "mask": mask,
            "bbox": (x_min, y_min, x_max, y_max),
            "area": area,
        })

    # Remove inner shapes from outer masks
    # Sort by area descending so outers are processed first
    mask_records.sort(key=lambda m: m["area"], reverse=True)

    for i, outer in enumerate(mask_records):
        x_min_o, y_min_o, x_max_o, y_max_o = outer["bbox"]
        for j in range(i + 1, len(mask_records)):
            inner = mask_records[j]
            x_min_i, y_min_i, x_max_i, y_max_i = inner["bbox"]

            # Quick bbox containment check
            if (x_min_o <= x_min_i <= x_max_i <= x_max_o) and (y_min_o <= y_min_i <= y_max_i <= y_max_o):
                # Subtract inner mask from outer mask
                outer["mask"] = np.clip(outer["mask"] - inner["mask"], 0, 1).astype(np.uint8)

    # Process masks into cropped images with metadata
    segments_data: List[Dict[str, Any]] = []
    crop_data: List[Dict[str, Any]] = []

    b, g, r = cv2.split(image)
    for record in mask_records:
        mask = record["mask"]
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            continue

        y_min, y_max = int(y_indices.min()), int(y_indices.max())
        x_min, x_max = int(x_indices.min()), int(x_indices.max())

        h_crop = y_max - y_min + 1
        w_crop = x_max - x_min + 1

        alpha = mask * 255
        rgba_image = cv2.merge([b, g, r, alpha])
        final_crop = rgba_image[y_min:y_max + 1, x_min:x_max + 1]

        image_base64 = encode_image_to_base64(final_crop)

        center_x = int(x_min + w_crop / 2)
        center_y = int(y_min + h_crop / 2)

        segments_data.append({
            "id": record["id"],
            "image": image_base64,
            "filename": f"obj_{record['id']}.png",
            "x": x_min,
            "y": y_min,
            "width": w_crop,
            "height": h_crop,
            "center_x": center_x,
            "center_y": center_y
        })

        if include_crops:
            crop_data.append({
                "id": record["id"],
                "image": final_crop,
                "width": w_crop,
                "height": h_crop,
            })

    logger.info(f"Generated {len(segments_data)} cropped segment images")
    return segments_data, crop_data, (w, h)


def _alpha_composite(bg: np.ndarray, fg: np.ndarray, x: int, y: int) -> None:
    """Alpha composite a BGRA foreground onto a BGRA background in-place."""
    h, w = fg.shape[:2]
    bg_region = bg[y:y + h, x:x + w]

    fg_alpha = fg[..., 3:4].astype(np.float32) / 255.0
    bg_alpha = bg_region[..., 3:4].astype(np.float32) / 255.0

    out_rgb = fg[..., :3].astype(np.float32) * fg_alpha + bg_region[..., :3].astype(np.float32) * (1.0 - fg_alpha)
    out_alpha = fg_alpha + bg_alpha * (1.0 - fg_alpha)

    bg_region[..., :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    bg_region[..., 3:4] = np.clip(out_alpha * 255.0, 0, 255).astype(np.uint8)


def _compose_jigsaw(
    crop_data: List[Dict[str, Any]],
    canvas_width: int,
    canvas_height: Optional[int],
    padding: int,
    background: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if canvas_width <= 0:
        raise ValueError("canvas_width must be > 0")
    if canvas_height is not None and canvas_height <= 0:
        raise ValueError("canvas_height must be > 0 when provided")

    placements: List[Dict[str, Any]] = []

    x_cursor = padding
    y_cursor = padding
    row_height = 0

    # Larger pieces first tends to pack better for a simple row layout
    sorted_crops = sorted(crop_data, key=lambda c: (c["height"] * c["width"]), reverse=True)

    for crop in sorted_crops:
        w_crop = crop["width"]
        h_crop = crop["height"]

        if x_cursor + w_crop + padding > canvas_width:
            x_cursor = padding
            y_cursor += row_height + padding
            row_height = 0

        placements.append({
            "id": crop["id"],
            "x": x_cursor,
            "y": y_cursor,
            "width": w_crop,
            "height": h_crop
        })

        x_cursor += w_crop + padding
        row_height = max(row_height, h_crop)

    required_height = y_cursor + row_height + padding
    final_height = required_height if canvas_height is None else max(canvas_height, required_height)

    canvas = np.zeros((final_height, canvas_width, 4), dtype=np.uint8)
    canvas[:, :] = np.array(background, dtype=np.uint8)

    for crop, placement in zip(sorted_crops, placements):
        _alpha_composite(canvas, crop["image"], placement["x"], placement["y"])

    return canvas, placements


def segment_image(image_bytes: bytes) -> List[Dict[str, Any]]:
    segments_data, _, _ = _extract_segments(image_bytes, include_crops=False)
    print("Done")
    return segments_data


def segment_image_with_puzzle(
    image_bytes: bytes,
    canvas_width: Optional[int] = None,
    canvas_height: Optional[int] = None,
    padding: int = 8,
    background: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> Dict[str, Any]:
    """
    Segment the image into separate objects and compose them into a jigsaw-like canvas.
    Returns both the cropped object images and the composed puzzle image.
    """
    segments_data, crop_data, (orig_w, orig_h) = _extract_segments(image_bytes, include_crops=True)

    if not crop_data:
        return {
            "segments": segments_data,
            "puzzle_image": None,
            "puzzle_layout": [],
            "puzzle_canvas": {"width": canvas_width or orig_w, "height": canvas_height or orig_h}
        }

    use_canvas_width = canvas_width or orig_w
    use_canvas_height = canvas_height or orig_h

    puzzle_canvas, placements = _compose_jigsaw(
        crop_data=crop_data,
        canvas_width=use_canvas_width,
        canvas_height=use_canvas_height,
        padding=padding,
        background=background
    )

    puzzle_base64 = encode_image_to_base64(puzzle_canvas)

    return {
        "segments": segments_data,
        "puzzle_image": puzzle_base64,
        "puzzle_layout": placements,
        "puzzle_canvas": {"width": puzzle_canvas.shape[1], "height": puzzle_canvas.shape[0]}
    }
