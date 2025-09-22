#!/usr/bin/env python3
"""
Extract Metadata from Processed JPG

This script extracts and displays metadata from processed JPG files,
focusing on EXIF data.
"""

import sys
from pathlib import Path
from PIL import Image
import piexif
import piexif.helper

def format_value(value):
    """Format EXIF values for display."""
    if isinstance(value, bytes):
        # Try to decode bytes, otherwise show as hex
        try:
            # For user comments and other text fields
            if value.startswith(b'UNICODE\x00'):
                return value[8:].decode('utf-16')
            return value.decode('utf-8', errors='ignore').strip('\x00')
        except:
            return value.hex()
    if isinstance(value, tuple) and len(value) == 2:
        # Rational numbers (like aperture, exposure time)
        if value[1] != 0:
            return f"{value[0]}/{value[1]} ({value[0]/value[1]:.4f})"
        return f"{value[0]}/{value[1]}"
    return value

def extract_jpg_metadata(image_path):
    """Extract and display metadata from a processed JPG file."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"Error: File {image_path} does not exist.")
        return
    
    try:
        with Image.open(image_path) as img:
            print(f"File Name: {image_path.name}")
            print(f"File Size: {image_path.stat().st_size} bytes ({image_path.stat().st_size/1024:.1f} KB)")
            print(f"Image Size: {img.size[0]} x {img.size[1]}")
            print(f"Format: {img.format}")
            print(f"Color Mode: {img.mode}")
            print()

            # Load EXIF data using piexif
            try:
                exif_data = piexif.load(str(image_path))
                
                # Display UserComment if it exists
                if piexif.ExifIFD.UserComment in exif_data.get("Exif", {}):
                    user_comment = piexif.helper.UserComment.load(exif_data["Exif"][piexif.ExifIFD.UserComment])
                    print(f"User Comment: {user_comment}")
                    print("-" * 40)

                # Iterate through the different EXIF sections (IFDs)
                for ifd_name in sorted(exif_data.keys()):
                    if ifd_name == "thumbnail":
                        print("Thumbnail: Present" if exif_data[ifd_name] else "Thumbnail: None")
                        continue
                    
                    print(f"--- {ifd_name} IFD ---")
                    
                    # Sort tags for consistent output
                    sorted_tags = sorted(exif_data[ifd_name].keys())
                    
                    for tag in sorted_tags:
                        tag_name = piexif.TAGS.get(ifd_name, {}).get(tag, {}).get("name", "Unknown")
                        value = exif_data[ifd_name][tag]
                        
                        # Truncate very long values
                        formatted_val = format_value(value)
                        if isinstance(formatted_val, str) and len(formatted_val) > 100:
                            formatted_val = formatted_val[:97] + "..."
                            
                        print(f"{tag_name:25} (0x{tag:04x}): {formatted_val}")
                    print()

            except Exception as e:
                print(f"Could not read EXIF data: {e}")

    except Exception as e:
        print(f"Error reading {image_path}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 extract_metadata.py <jpg_file>")
        print("Example: python3 extract_metadata.py processing/photo_processed.jpg")
        sys.exit(1)
    
    extract_jpg_metadata(sys.argv[1])

if __name__ == "__main__":
    main()