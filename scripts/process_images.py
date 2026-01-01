import os
import argparse
import glob
import time
import numpy as np
import cv2
import onnxruntime
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Configuration
MODEL_REPO = "skytnt/anime-seg"
MODEL_FILENAME = "isnetis.onnx"
INPUT_DIR = "costume_origin"
OUTPUT_DIR = "costume_icons"
MAX_SIZE = 512
MODEL_SIZE = 1024
HOURS = int(os.environ.get("HOURS", "168"))

def get_recent_files(directory, hours):
    cutoff_time = time.time() - (hours * 3600)
    files = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []

    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_extensions):
            filepath = os.path.join(directory, filename)
            if os.path.getmtime(filepath) >= cutoff_time:
                files.append(filepath)
    return files

def get_mask(session, input_img, size=1024):
    # Preprocessing: Letterbox resize to 1024x1024
    h0, w0 = input_img.shape[:2]
    
    # Calculate scaling to fit within size x size while maintaining aspect ratio
    scale = min(size / h0, size / w0)
    h, w = int(h0 * scale), int(w0 * scale)
    
    # Resize image
    img_resized = cv2.resize(input_img, (w, h))
    
    # Calculate padding
    ph, pw = size - h, size - w
    
    # Create canvas with padding (centered)
    # SkyTNT implementation centers it
    img_input = np.zeros([size, size, 3], dtype=np.float32)
    # Top/Left padding
    pt = ph // 2
    pl = pw // 2
    img_input[pt:pt+h, pl:pl+w] = img_resized
    
    # Normalize: / 255.0 (No mean/std subtraction in SkyTNT's ONNX export usually, checking inference.py it says / 255)
    img_input = img_input / 255.0
    
    # HWC -> CHW
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :] # Batch dimension
    
    # Inference
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: img_input.astype(np.float32)}
    pred = session.run(None, ort_inputs)[0]
    
    # Postprocessing
    pred = pred[0] # remove batch
    pred = np.transpose(pred, (1, 2, 0)) # CHW -> HWC
    
    # Crop back to content
    pred = pred[pt:pt+h, pl:pl+w]
    
    # Resize back to original size
    pred = cv2.resize(pred, (w0, h0))
    # Add axis to make it (H, W, 1)
    if len(pred.shape) == 2:
        pred = pred[:, :, np.newaxis]
        
    return pred

def process_images():
    print(f"🚀 Starting processing (Last {HOURS} hours)")
    
    files = get_recent_files(INPUT_DIR, HOURS)
    if not files:
        print("📭 No new files found.")
        return

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download model if needed
    print(f"🤖 Loading model {MODEL_FILENAME}...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    
    # Initialize ONNX Runtime
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    for filepath in tqdm(files):
        filename = os.path.basename(filepath)
        print(f"📷 Processing: {filename}")
        
        try:
            # Read image
            # Handle non-ascii paths by using numpy fromfile if needed, but standard cv2.imread might fail on Windows with unicode
            # Using numpy method to be safe
            img_stream = np.fromfile(filepath, dtype=np.uint8)
            img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  ❌ Failed to read image: {filepath}")
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get Mask
            mask = get_mask(session, img_rgb, size=MODEL_SIZE)
            
            # Composite
            # mask is 0..1
            
            # Apply mask to original image
            # Start with transparent background
            h, w = img_rgb.shape[:2]
            
            # Ensure mask is single channel
            if mask.shape[2] != 1:
                mask = mask[:, :, 0:1]

            # Create RGBA
            r, g, b = cv2.split(img_rgb)
            a = (mask * 255).astype(np.uint8)
            
            rgba_image = cv2.merge([r, g, b, a])
            
            # Trim transparent edges (equivalent to sharp's trim)
            # Find bounding box of non-transparent area
            non_zero = cv2.findNonZero(a)
            if non_zero is not None:
                x, y, w_box, h_box = cv2.boundingRect(non_zero)
                # Add a small padding or threshold if needed, but boundingRect is exact non-zero
                cropped_rgba = rgba_image[y:y+h_box, x:x+w_box]
            else:
                cropped_rgba = rgba_image
                
            # Resize fit inside MAX_SIZE x MAX_SIZE without enlargement
            h_c, w_c = cropped_rgba.shape[:2]
            if h_c > MAX_SIZE or w_c > MAX_SIZE:
                scale = min(MAX_SIZE / h_c, MAX_SIZE / w_c)
                new_h, new_w = int(h_c * scale), int(w_c * scale)
                final_img = cv2.resize(cropped_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                final_img = cropped_rgba

            # Save
            basename = os.path.splitext(filename)[0]
            output_path = os.path.join(OUTPUT_DIR, f"{basename}.png")
            
            # Save using imencode for unicode support
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(final_img, cv2.COLOR_RGBA2BGRA))
            if is_success:
                with open(output_path, "wb") as f:
                    f.write(buffer)
                print(f"  ✅ Saved: {basename}.png")
            else:
                print(f"  ❌ Failed to encode png: {basename}")
                
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    process_images()
