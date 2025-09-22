#!/usr/bin/env python3
"""
Image Processing Script

This script processes images from the incoming directory by:
1. Resizing them to a fixed size (1024x768)
2. Reducing file size by adjusting JPEG quality to meet a target
3. Preserving EXIF metadata
4. Saving processed images to the processing directory as JPG

Usage:
    python3 process_image.py [filename]
    
If no filename is provided, it processes all images in the incoming directory.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ExifTags
import piexif
import json

# Directory paths - using relative paths for better portability
SCRIPT_DIR = Path(__file__).parent.resolve()
INCOMING_DIR = SCRIPT_DIR / 'incoming'
PROCESSING_DIR = SCRIPT_DIR / 'processing'
ARCHIVED_DIR = SCRIPT_DIR / 'archived'
INTERSECTION_DATA_PATH = PROCESSING_DIR / 'intersection_data.json'

# Ensure directories exist
INCOMING_DIR.mkdir(exist_ok=True)
PROCESSING_DIR.mkdir(exist_ok=True)
ARCHIVED_DIR.mkdir(exist_ok=True)

def get_image_metadata(image_path):
    """Extract EXIF metadata from a JPEG image."""
    try:
        exif_dict = piexif.load(str(image_path))
        return exif_dict
    except Exception as e:
        print(f"  Warning: Could not extract EXIF data from {image_path.name}: {e}")
        return None

def get_font(font_size):
    """Attempt to load a Courier font, falling back to a default."""
    font_paths = [
        "/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf",
        "/usr/share/fonts/corefonts/cour.ttf",  # Common on Linux
        "C:\\Windows\\Fonts\\cour.ttf",  # Windows
        "/System/Library/Fonts/Courier.dfont"  # macOS
    ]
    for path in font_paths:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    # Fallback to Pillow's default font if Courier is not found
    print("  Warning: Courier font not found. Using default font.")
    return ImageFont.load_default()

def create_watermark_text(exif_data, intersection_data):
    """Create the multi-line watermark text."""
    # Extract timestamp from EXIF
    timestamp_str = "Timestamp not available"
    if exif_data and "Exif" in exif_data and piexif.ExifIFD.DateTimeOriginal in exif_data["Exif"]:
        try:
            timestamp_bytes = exif_data["Exif"][piexif.ExifIFD.DateTimeOriginal]
            timestamp_str = timestamp_bytes.decode('utf-8')
        except Exception as e:
            print(f"  Warning: Could not decode timestamp: {e}")

    watermark_lines = [timestamp_str, ""]  # Add a blank line

    # Add key-value pairs from intersection data
    for key, value in intersection_data.items():
        watermark_lines.append(f"{key}: {value}")
        
    return "\n".join(watermark_lines)

def resize_and_compress_jpg(input_path, output_path, target_size_kb=100, target_dims=(1024, 768)):
    """
    Resize, compress, watermark, and save a JPG image to meet a target file size.
    
    Args:
        input_path (Path): Path to the input image.
        output_path (Path): Path to save the processed image.
        target_size_kb (int): The target file size in kilobytes.
        target_dims (tuple): The target dimensions (width, height).
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Load intersection data for watermark
        intersection_data = {}
        if INTERSECTION_DATA_PATH.exists():
            with open(INTERSECTION_DATA_PATH, 'r') as f:
                intersection_data = json.load(f)
        else:
            print(f"  Warning: {INTERSECTION_DATA_PATH} not found. Watermark will be incomplete.")

        with Image.open(input_path) as img:
            # Preserve original EXIF data
            metadata = get_image_metadata(input_path)
            exif_bytes = b''
            if metadata:
                try:
                    exif_bytes = piexif.dump(metadata)
                except Exception as e:
                    print(f"  Warning: Could not dump EXIF data: {e}")

            # Resize the image, ignoring original aspect ratio as requested
            resized_img = img.resize(target_dims, Image.Resampling.LANCZOS)
            
            # Convert to RGB if it's not already
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')

            # --- Add Watermark ---
            # Create a semi-transparent background for the text
            draw = ImageDraw.Draw(resized_img, "RGBA")
            font = get_font(48) # Increased font size
            watermark_text = create_watermark_text(metadata, intersection_data)
            
            # Define line spacing
            line_spacing = 15

            text_bbox = draw.multiline_textbbox((0, 0), watermark_text, font=font, spacing=line_spacing)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            margin = 30 # Increased margin
            position = (margin, resized_img.height - text_height - margin)
            
            # Create a rectangle for the background
            bg_position = (
                position[0] - 15, 
                position[1] - 15, 
                position[0] + text_width + 15, 
                position[1] + text_height + 15
            )
            # Draw the semi-transparent rectangle
            draw.rectangle(bg_position, fill=(0, 0, 0, 128)) # Black with 50% opacity

            # Draw the text on top of the background
            draw.multiline_text(position, watermark_text, font=font, fill=(50, 205, 50), spacing=line_spacing) # Lime Green

            # Iteratively adjust JPEG quality to meet the file size target
            quality = 85  # Start with a high quality
            
            while quality > 10:
                resized_img.save(output_path, "JPEG", quality=quality, exif=exif_bytes, optimize=True, dpi=(72, 72))
                
                file_size_kb = output_path.stat().st_size / 1024
                
                if file_size_kb <= target_size_kb:
                    print(f"  Achieved target size: {file_size_kb:.1f} KB with quality={quality}")
                    return True
                
                # Reduce quality for the next attempt
                if file_size_kb > target_size_kb * 1.5:
                    quality -= 10  # Larger jump if far from target
                else:
                    quality -= 5   # Smaller jump if close
            
            print(f"  Warning: Could not achieve target size. Final size: {output_path.stat().st_size / 1024:.1f} KB")
            return True

    except Exception as e:
        print(f"Error during image processing for {input_path.name}: {e}")
        return False

def process_single_image(input_path, target_size_kb=100):
    """
    Process a single image file.
    
    Args:
        input_path (Path): Path to input image.
        target_size_kb (int): Target file size in KB.
    
    Returns:
        Path: Path to processed image or None if failed.
    """
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist.")
        return None
    
    print(f"Processing: {input_path.name}")
    
    original_size = input_path.stat().st_size
    with Image.open(input_path) as img:
        original_dimensions = img.size
    
    print(f"  Original: {original_dimensions[0]}x{original_dimensions[1]}, {original_size/1024:.1f} KB")
    
    # Generate output filename
    output_filename = input_path.stem + "_processed.jpg"
    output_path = PROCESSING_DIR / output_filename
    
    success = resize_and_compress_jpg(input_path, output_path, target_size_kb)
    
    if success:
        final_size = output_path.stat().st_size
        size_reduction = ((original_size - final_size) / original_size * 100) if original_size > 0 else 0
        
        print(f"  Saved to: {output_path}")
        print(f"  Size reduction: {size_reduction:.1f}%")
        return output_path
    else:
        print(f"  Failed to process {input_path.name}")
        return None

def process_all_images(target_size_kb=45):
    """Process all JPG images in the incoming directory."""
    
    image_extensions = {'.jpg', '.jpeg'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(INCOMING_DIR.glob(f"*{ext}"))
        image_files.extend(INCOMING_DIR.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("No JPG files found in the incoming directory.")
        return []
    
    print(f"Found {len(image_files)} image(s) to process:")
    
    processed_files = []
    for image_file in image_files:
        result = process_single_image(image_file, target_size_kb)
        if result:
            processed_files.append(image_file) # Keep track of original files
        print()
    
    print(f"Successfully processed {len(processed_files)} out of {len(image_files)} images.")
    return processed_files

def archive_processed_images(original_files):
    """Move original images to the archived directory."""
    if not original_files:
        return
        
    print("\nArchiving original files...")
    for original_path in original_files:
        if original_path.exists():
            archive_path = ARCHIVED_DIR / original_path.name
            try:
                original_path.rename(archive_path)
                print(f"  Archived: {original_path.name} -> {archive_path}")
            except Exception as e:
                print(f"  Error archiving {original_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process JPG images: resize, compress, watermark, and preserve EXIF.")
    parser.add_argument('filename', nargs='?', help='Specific file to process in the incoming directory (optional).')
    parser.add_argument('--size', type=int, default=100, help='Target file size in KB (default: 100).')
    parser.add_argument('--archive', action='store_true', help='Archive original files after processing.')
    
    args = parser.parse_args()
    
    print("Image Processing Script")
    print("=" * 50)
    print(f"Incoming directory:   {INCOMING_DIR}")
    print(f"Processing directory: {PROCESSING_DIR}")
    print(f"Archive directory:    {ARCHIVED_DIR}")
    print(f"Target size:          {args.size} KB")
    print()
    
    processed_originals = []
    
    if args.filename:
        input_path = INCOMING_DIR / args.filename
        result = process_single_image(input_path, args.size)
        if result:
            processed_originals.append(input_path)
    else:
        processed_originals = process_all_images(args.size)
        
    if args.archive and processed_originals:
        archive_processed_images(processed_originals)

if __name__ == "__main__":
    main()