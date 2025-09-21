# Image Processing Scripts

This directory contains scripts for processing images by resizing them to reduce file size while preserving metadata and converting to PNG format.

## Files

- `process_image.py`: Main Python script for image processing
- `process_images.sh`: Bash utility script for easy batch processing
- `incoming/`: Directory containing images to be processed
- `processing/`: Directory where processed images are saved
- `archived/`: Directory for storing original images after processing

## Features

- **Size Reduction**: Intelligently resizes images to target file size (default: 500KB)
- **Metadata Preservation**: Keeps all EXIF data and other metadata intact
- **PNG Conversion**: Converts images to PNG format while preserving quality
- **Batch Processing**: Can process individual files or entire directories
- **Archiving**: Option to move original files to archived directory

## Usage

### Using the Python Script Directly

```bash
# Process a specific image
python3 process_image.py PXL_20250916_003238285.jpg

# Process all images in incoming directory
python3 process_image.py

# Process with custom target size (300KB)
python3 process_image.py --size 300

# Process and archive original files
python3 process_image.py --archive

# Process specific file with custom size and archive
python3 process_image.py PXL_20250916_003238285.jpg --size 300 --archive
```

### Using the Bash Utility Script

```bash
# Process all images
./process_images.sh

# Process a specific image
./process_images.sh photo.jpg

# Process with custom target size
./process_images.sh size 300

# Process and archive originals
./process_images.sh archive
```

## How It Works

1. **Metadata Extraction**: Reads all EXIF and other metadata from the original image
2. **Smart Resizing**: Calculates optimal dimensions to achieve target file size while maintaining aspect ratio
3. **High-Quality Resampling**: Uses Lanczos resampling for best quality during resize
4. **Metadata Transfer**: Embeds original metadata into the PNG file
5. **Format Conversion**: Saves as optimized PNG with preserved metadata

## Example Output

```
Processing: PXL_20250916_003238285.jpg
  Original: 8160x6144, 9076.3 KB
  Processed: 816x614, 694.4 KB
  Saved to: /home/kb1hgo/image/processing/PXL_20250916_003238285_processed.png
  Size reduction: 92.3%
```

## Requirements

- Python 3.x
- Pillow (PIL) library
- piexif library (for advanced EXIF handling)

Both libraries are already installed in this environment.

## Directory Structure

```
/home/kb1hgo/image/
├── process_image.py          # Main processing script
├── process_images.sh         # Bash utility script
├── README.md                # This documentation
├── incoming/                # Place images here for processing
├── processing/              # Processed images appear here
└── archived/                # Original images moved here when archived
```

## Notes

- The script preserves image quality while significantly reducing file size
- Transparency is maintained for images that support it
- The target size is approximate - actual size may vary slightly
- Original images remain in incoming/ unless --archive option is used
- All processed images are saved with "_processed.png" suffix