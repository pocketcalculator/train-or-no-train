#!/bin/bash
# 
# Image Processing Utility Script
# 
# This script provides easy commands for processing images:
# - process_images: Process all images in incoming directory
# - process_image <filename>: Process a specific image
# - archive_originals: Move processed originals to archived directory
#

SCRIPT_DIR="/home/kb1hgo/image"
PROCESS_SCRIPT="$SCRIPT_DIR/process_image.py"

# Function to process all images
process_images() {
    echo "Processing all images in incoming directory..."
    cd "$SCRIPT_DIR" && python3 "$PROCESS_SCRIPT"
}

# Function to process a specific image
process_image() {
    if [ -z "$1" ]; then
        echo "Usage: process_image <filename>"
        echo "Example: process_image photo.jpg"
        return 1
    fi
    
    echo "Processing image: $1"
    cd "$SCRIPT_DIR" && python3 "$PROCESS_SCRIPT" "$1"
}

# Function to process with custom size
process_images_size() {
    if [ -z "$1" ]; then
        echo "Usage: process_images_size <size_in_kb>"
        echo "Example: process_images_size 300"
        return 1
    fi
    
    echo "Processing all images with target size: $1 KB"
    cd "$SCRIPT_DIR" && python3 "$PROCESS_SCRIPT" --size "$1"
}

# Function to process and archive
process_and_archive() {
    echo "Processing all images and archiving originals..."
    cd "$SCRIPT_DIR" && python3 "$PROCESS_SCRIPT" --archive
}

# Main script logic
case "$1" in
    "all"|"")
        process_images
        ;;
    "size")
        process_images_size "$2"
        ;;
    "archive")
        process_and_archive
        ;;
    *)
        process_image "$1"
        ;;
esac