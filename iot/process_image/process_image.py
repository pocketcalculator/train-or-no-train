#!/usr/bin/env python3
"""
Image Processing Script

This script processes images from the incoming directory by:
1. Resizing them to reduce file size while maintaining quality
2. Preserving all metadata (EXIF, IPTC, etc.)
3. Converting to PNG format
4. Saving processed images to the processing directory

Usage:
    python3 process_image.py [filename]
    
If no filename is provided, it processes all images in the incoming directory.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
import piexif
import json

# Directory paths
INCOMING_DIR = Path('/home/kb1hgo/image/incoming')
PROCESSING_DIR = Path('/home/kb1hgo/image/processing')
ARCHIVED_DIR = Path('/home/kb1hgo/image/archived')

# Ensure processing directory exists
PROCESSING_DIR.mkdir(exist_ok=True)
ARCHIVED_DIR.mkdir(exist_ok=True)

def get_image_metadata(image_path):
    """Extract all metadata from an image."""
    metadata = {}
    
    try:
        with Image.open(image_path) as img:
            # Get EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                exif_dict = {}
                exif = img._getexif()
                for tag, value in exif.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    exif_dict[decoded] = value
                metadata['exif'] = exif_dict
            
            # Get raw EXIF for preservation
            if 'exif' in img.info:
                metadata['raw_exif'] = img.info['exif']
            
            # Get other metadata
            for key, value in img.info.items():
                if key not in ['exif']:
                    metadata[key] = value
                    
    except Exception as e:
        print(f"Warning: Could not extract metadata from {image_path}: {e}")
    
    return metadata

def resize_image_smart(image, target_size_kb=500, quality_start=85):
    """
    Resize image to target file size while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        target_size_kb: Target file size in KB
        quality_start: Starting quality for JPEG compression
    
    Returns:
        Resized PIL Image object
    """
    # Calculate scaling factor based on target size
    original_width, original_height = image.size
    original_pixels = original_width * original_height
    
    # Estimate scaling factor (rough approximation)
    # Assume roughly 3 bytes per pixel for RGB
    estimated_size_kb = (original_pixels * 3) / 1024
    
    if estimated_size_kb <= target_size_kb:
        return image  # Image is already small enough
    
    # Calculate scale factor to reduce size
    scale_factor = (target_size_kb / estimated_size_kb) ** 0.5
    
    # Ensure minimum reasonable size
    scale_factor = max(scale_factor, 0.1)
    
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Use high-quality resampling
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized

def transfer_metadata_to_png(metadata, png_info=None):
    """Transfer metadata to PNG format."""
    if png_info is None:
        png_info = PngInfo()
    
    # Add text metadata
    for key, value in metadata.items():
        if key == 'exif' and isinstance(value, dict):
            # Convert EXIF dict to JSON string for PNG
            png_info.add_text("EXIF", json.dumps(value, default=str))
        elif key == 'raw_exif':
            # Try to preserve raw EXIF data as a text chunk
            try:
                # Convert raw EXIF bytes to base64 for storage in PNG
                import base64
                exif_b64 = base64.b64encode(value).decode('ascii')
                png_info.add_text("Raw_EXIF", exif_b64)
            except Exception as e:
                print(f"  Warning: Could not preserve raw EXIF: {e}")
        elif isinstance(value, (str, int, float)):
            png_info.add_text(str(key), str(value))
    
    return png_info

def process_single_image(input_path, target_size_kb=500):
    """
    Process a single image file.
    
    Args:
        input_path: Path to input image
        target_size_kb: Target file size in KB
    
    Returns:
        Path to processed image or None if failed
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist.")
        return None
    
    print(f"Processing: {input_path.name}")
    
    try:
        # Extract metadata before opening for processing
        metadata = get_image_metadata(input_path)
        
        # Open and process image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (for PNG compatibility)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Keep transparency for RGBA, convert others to RGB
                if img.mode == 'P' and 'transparency' in img.info:
                    img = img.convert('RGBA')
                elif img.mode != 'RGBA':
                    img = img.convert('RGB')
            elif img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Get original size info
            original_size = input_path.stat().st_size
            original_dimensions = img.size
            
            print(f"  Original: {original_dimensions[0]}x{original_dimensions[1]}, {original_size/1024:.1f} KB")
            
            # Resize image
            processed_img = resize_image_smart(img, target_size_kb)
            new_dimensions = processed_img.size
            
            # Prepare PNG metadata
            png_info = transfer_metadata_to_png(metadata)
            
            # Generate output filename
            output_filename = input_path.stem + "_processed.png"
            output_path = PROCESSING_DIR / output_filename
            
            # Save as PNG with metadata
            processed_img.save(output_path, "PNG", pnginfo=png_info, optimize=True)
            
            # Get final size
            final_size = output_path.stat().st_size
            
            print(f"  Processed: {new_dimensions[0]}x{new_dimensions[1]}, {final_size/1024:.1f} KB")
            print(f"  Saved to: {output_path}")
            print(f"  Size reduction: {((original_size - final_size) / original_size * 100):.1f}%")
            
            return output_path
            
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None

def process_all_images(target_size_kb=500):
    """Process all images in the incoming directory."""
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(INCOMING_DIR.glob(f"*{ext}"))
        image_files.extend(INCOMING_DIR.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("No image files found in the incoming directory.")
        return
    
    print(f"Found {len(image_files)} image(s) to process:")
    
    processed_count = 0
    for image_file in image_files:
        result = process_single_image(image_file, target_size_kb)
        if result:
            processed_count += 1
            print()
    
    print(f"Successfully processed {processed_count} out of {len(image_files)} images.")

def archive_processed_image(original_path):
    """Move original image to archived directory."""
    original_path = Path(original_path)
    if original_path.exists():
        archive_path = ARCHIVED_DIR / original_path.name
        original_path.rename(archive_path)
        print(f"Archived original to: {archive_path}")

def main():
    parser = argparse.ArgumentParser(description="Process images: resize, preserve metadata, convert to PNG")
    parser.add_argument('filename', nargs='?', help='Specific file to process (optional)')
    parser.add_argument('--size', type=int, default=500, help='Target file size in KB (default: 500)')
    parser.add_argument('--archive', action='store_true', help='Archive original files after processing')
    
    args = parser.parse_args()
    
    print("Image Processing Script")
    print("=" * 50)
    print(f"Incoming directory: {INCOMING_DIR}")
    print(f"Processing directory: {PROCESSING_DIR}")
    print(f"Target size: {args.size} KB")
    print()
    
    if args.filename:
        # Process specific file
        input_path = INCOMING_DIR / args.filename
        result = process_single_image(input_path, args.size)
        
        if result and args.archive:
            archive_processed_image(input_path)
    else:
        # Process all files
        process_all_images(args.size)
        
        if args.archive:
            # Ask for confirmation before archiving all
            response = input("\nArchive all original files? (y/N): ")
            if response.lower() in ['y', 'yes']:
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
                for ext in image_extensions:
                    for image_file in INCOMING_DIR.glob(f"*{ext}"):
                        archive_processed_image(image_file)
                    for image_file in INCOMING_DIR.glob(f"*{ext.upper()}"):
                        archive_processed_image(image_file)

if __name__ == "__main__":
    main()