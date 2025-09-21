#!/usr/bin/env python3
"""
Extract Metadata from Processed PNG

This script extracts and displays metadata from processed PNG files
in a format similar to exiftool output.
"""

import sys
import json
import base64
from pathlib import Path
from PIL import Image

def extract_png_metadata(image_path):
    """Extract and display metadata from a processed PNG file."""
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
            
            # Extract EXIF data from JSON
            if 'EXIF' in img.info:
                try:
                    exif_data = json.loads(img.info['EXIF'])
                    print("EXIF Data (from JSON):")
                    print("-" * 40)
                    
                    # Sort and display EXIF data
                    for key, value in sorted(exif_data.items()):
                        if key == 'GPSInfo' and isinstance(value, dict):
                            print(f"{key:20}: GPS coordinates available")
                            # Show GPS details if available
                            if '2' in value and '4' in value:  # Latitude and Longitude
                                lat = value.get('2', 'N/A')
                                lon = value.get('4', 'N/A')
                                lat_ref = value.get('1', '')
                                lon_ref = value.get('3', '')
                                print(f"{'GPS Latitude':20}: {lat} {lat_ref}")
                                print(f"{'GPS Longitude':20}: {lon} {lon_ref}")
                        elif isinstance(value, (str, int, float)):
                            # Truncate very long values
                            if isinstance(value, str) and len(value) > 50:
                                print(f"{key:20}: {value[:47]}...")
                            else:
                                print(f"{key:20}: {value}")
                        elif isinstance(value, list) and len(value) <= 3:
                            print(f"{key:20}: {value}")
                        else:
                            print(f"{key:20}: [Complex data]")
                    
                    print()
                except json.JSONDecodeError:
                    print("EXIF data present but not valid JSON")
            
            # Show other metadata
            other_data = {k: v for k, v in img.info.items() if k not in ['EXIF', 'Raw_EXIF', 'icc_profile']}
            if other_data:
                print("Other Metadata:")
                print("-" * 40)
                for key, value in other_data.items():
                    print(f"{key:20}: {value}")
                print()
            
            # Show technical info
            if 'icc_profile' in img.info:
                print(f"ICC Profile: Present ({len(img.info['icc_profile'])} bytes)")
            
            if 'Raw_EXIF' in img.info:
                try:
                    raw_exif = base64.b64decode(img.info['Raw_EXIF'])
                    print(f"Raw EXIF Data: Present ({len(raw_exif)} bytes)")
                except:
                    print("Raw EXIF Data: Present but decode failed")
                    
    except Exception as e:
        print(f"Error reading {image_path}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 extract_metadata.py <png_file>")
        print("Example: python3 extract_metadata.py processing/photo_processed.png")
        sys.exit(1)
    
    extract_png_metadata(sys.argv[1])

if __name__ == "__main__":
    main()