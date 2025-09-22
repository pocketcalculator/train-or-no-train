# Image Processing Script

This directory contains a Python script to process images, primarily for preparing them for use in object detection models or for consistent archiving. The script resizes, compresses, and watermarks images.

## Files

- `process_image.py`: The main Python script for image processing.
- `process_images.sh`: A bash utility script for easier command-line usage.
- `README.md`: This documentation.
- `incoming/`: Directory where you should place images to be processed.
- `processing/`: Directory where processed images are saved.
- `archived/`: Directory where original images are moved after processing (if the archive option is used).
- `processing/intersection_data.json`: (Optional) A JSON file containing key-value pairs to be added to the watermark.

## Features

- **Fixed Resize**: Resizes all images to a standard `1024x768` resolution, ignoring the original aspect ratio.
- **Targeted Compression**: Adjusts JPEG quality to bring the file size under a specified target (default: 100 KB).
- **Watermarking**: Adds a semi-transparent watermark containing the image timestamp (from EXIF data) and custom data from `processing/intersection_data.json`.
- **Metadata Preservation**: Preserves original EXIF metadata in the processed JPG files.
- **Batch Processing**: Can process a single image or all images in the `incoming` directory.
- **Archiving**: Includes an option to move original files to the `archived` directory after successful processing.

## Usage

### Using the Python Script (`process_image.py`)

The Python script is the core of the processing logic.

```bash
# Process a specific image with default settings (target size 100 KB)
python3 process_image.py my_photo.jpg

# Process all images in the 'incoming' directory
python3 process_image.py

# Process with a custom target size (e.g., 45 KB)
python3 process_image.py --size 45

# Process and archive the original files
python3 process_image.py --archive

# Combine options: process a specific file with a custom size and archive it
python3 process_image.py my_photo.jpg --size 45 --archive
```

### Using the Bash Utility (`process_images.sh`)

The bash script is a convenient wrapper for the Python script.

```bash
# Process all images in the 'incoming' directory
./process_images.sh

# Process a specific image
./process_images.sh my_photo.jpg

# Process all images with a custom target size (e.g., 45 KB)
./process_images.sh size 45

# Process all images and archive the originals
./process_images.sh archive
```

## How It Works

1.  **Load Image**: Opens the specified JPG image from the `incoming` directory.
2.  **Extract Metadata**: Reads the EXIF data from the original image to preserve it.
3.  **Resize**: Resizes the image to `1024x768` using a high-quality Lanczos filter.
4.  **Apply Watermark**:
    *   Reads the timestamp from the EXIF data.
    *   Loads key-value pairs from `processing/intersection_data.json` (if it exists).
    *   Draws a semi-transparent black background in the bottom-left corner.
    *   Draws the timestamp and JSON data as lime green text on the background.
5.  **Compress**: Iteratively saves the image with decreasing JPEG quality until the file size is below the target.
6.  **Save Image**: Saves the final processed image as a JPG file in the `processing` directory, with `_processed.jpg` appended to the original filename.
7.  **Archive (Optional)**: If the `--archive` flag is used, the original image is moved from `incoming/` to `archived/`.

## Example Output

```
Processing: PXL_20250921_123456789.jpg
  Original: 4032x3024, 3450.2 KB
  Warning: intersection_data.json not found. Watermark will be incomplete.
  Achieved target size: 98.7 KB with quality=75
  Saved to: processing/PXL_20250921_123456789_processed.jpg
  Size reduction: 97.1%
```

## Requirements

- Python 3.x
- Pillow (`PIL`) library
- `piexif` library

## Directory Structure

```
.
├── process_image.py          # Main processing script
├── process_images.sh         # Bash utility script
├── README.md                 # This documentation
├── incoming/                 # Place images here for processing
│   └── my_photo.jpg
├── processing/               # Processed images appear here
│   ├── intersection_data.json  # (Optional) data for watermarks
│   └── my_photo_processed.jpg
└── archived/                 # Original images moved here when archived
    └── my_photo.jpg
```