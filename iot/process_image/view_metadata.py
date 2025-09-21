#!/usr/bin/env python3
"""
Metadata Viewer Script

This script displays metadata from both original and processed images
to help verify metadata preservation.
"""

import sys
import json
import base64
from pathlib import Path
from PIL import Image, ExifTags
import piexif

def display_image_metadata(image_path, title="Image Metadata"):
    """Display comprehensive metadata for an image."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"Error: File {image_path} does not exist.")
        return
    
    print(f"\n{'='*20} {title} {'='*20}")
    print(f"File: {image_path}")
    print(f"Size on disk: {image_path.stat().st_size / 1024:.1f} KB")
    
    try:
        with Image.open(image_path) as img:
            print(f"Format: {img.format}")
            print(f"Dimensions: {img.size[0]}x{img.size[1]}")
            print(f"Mode: {img.mode}")
            
            # For JPEG/original images
            if img.format in ['JPEG', 'MPO']:
                print("\n--- EXIF Data (via PIL) ---")
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    print(f"EXIF entries found: {len(exif)}")
                    
                    # Show key EXIF data
                    key_tags = ['Make', 'Model', 'DateTime', 'Software', 'Orientation', 'XResolution', 'YResolution']
                    for tag, value in exif.items():
                        tag_name = ExifTags.TAGS.get(tag, f"Tag_{tag}")
                        if tag_name in key_tags:
                            print(f"  {tag_name}: {value}")
                    
                    # Show GPS if available
                    gps_tag = None
                    for tag, value in exif.items():
                        if ExifTags.TAGS.get(tag) == 'GPSInfo':
                            gps_tag = value
                            break
                    
                    if gps_tag:
                        print(f"  GPS Info: Available ({len(gps_tag)} entries)")
                else:
                    print("No EXIF data found")
            
            # For PNG/processed images
            elif img.format == 'PNG':
                print("\n--- PNG Text Chunks ---")
                for key, value in img.info.items():
                    if key == 'EXIF':
                        try:
                            exif_data = json.loads(value)
                            print(f"  EXIF: Preserved as JSON ({len(exif_data)} entries)")
                            
                            # Show key preserved data
                            key_fields = ['Make', 'Model', 'DateTime', 'Software', 'Orientation']
                            for field in key_fields:
                                if field in exif_data:
                                    print(f"    {field}: {exif_data[field]}")
                            
                            if 'GPSInfo' in exif_data:
                                print(f"    GPS Info: Preserved")
                                
                        except json.JSONDecodeError:
                            print(f"  EXIF: Present but not valid JSON")
                    elif key == 'Raw_EXIF':
                        try:
                            raw_exif = base64.b64decode(value)
                            print(f"  Raw EXIF: Preserved ({len(raw_exif)} bytes)")
                        except:
                            print(f"  Raw EXIF: Present but decode failed")
                    elif key == 'icc_profile':
                        print(f"  ICC Profile: Preserved ({len(value)} bytes)")
                    else:
                        if isinstance(value, str) and len(value) > 50:
                            print(f"  {key}: {value[:50]}...")
                        else:
                            print(f"  {key}: {value}")
            
            print(f"\nAll info keys: {list(img.info.keys())}")
            
    except Exception as e:
        print(f"Error reading {image_path}: {e}")

def compare_metadata(original_path, processed_path):
    """Compare metadata between original and processed images."""
    print("\n" + "="*60)
    print("METADATA COMPARISON")
    print("="*60)
    
    display_image_metadata(original_path, "ORIGINAL IMAGE")
    display_image_metadata(processed_path, "PROCESSED IMAGE")
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    
    try:
        with Image.open(original_path) as orig:
            with Image.open(processed_path) as proc:
                orig_size = Path(original_path).stat().st_size
                proc_size = Path(processed_path).stat().st_size
                size_reduction = ((orig_size - proc_size) / orig_size) * 100
                
                print(f"File size reduction: {size_reduction:.1f}%")
                print(f"Dimension reduction: {orig.size} → {proc.size}")
                
                # Check metadata preservation
                orig_has_exif = hasattr(orig, '_getexif') and orig._getexif()
                proc_has_metadata = 'EXIF' in proc.info or 'Raw_EXIF' in proc.info
                
                if orig_has_exif and proc_has_metadata:
                    print("✅ Metadata preservation: SUCCESS")
                elif orig_has_exif:
                    print("❌ Metadata preservation: FAILED")
                else:
                    print("ℹ️  Original had no EXIF data")
                    
    except Exception as e:
        print(f"Error in comparison: {e}")

def main():
    if len(sys.argv) == 2:
        # Single image
        display_image_metadata(sys.argv[1])
    elif len(sys.argv) == 3:
        # Compare two images
        compare_metadata(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  python3 view_metadata.py <image_path>")
        print("  python3 view_metadata.py <original_path> <processed_path>")
        print("\nExamples:")
        print("  python3 view_metadata.py incoming/photo.jpg")
        print("  python3 view_metadata.py incoming/photo.jpg processing/photo_processed.png")

if __name__ == "__main__":
    main()