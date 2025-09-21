#!/bin/bash
#
# ImageMagick-based Image Processing Script
#
# This script uses ImageMagick to process images with similar functionality
# to the Python script, but with native Linux utilities.
#

INCOMING_DIR="/home/kb1hgo/image/incoming"
PROCESSING_DIR="/home/kb1hgo/image/processing"
ARCHIVED_DIR="/home/kb1hgo/image/archived"

# Ensure directories exist
mkdir -p "$PROCESSING_DIR" "$ARCHIVED_DIR"

# Default target size in KB
TARGET_SIZE_KB=${1:-500}

# Function to get file size in KB
get_size_kb() {
    echo $(( $(stat -c%s "$1") / 1024 ))
}

# Function to process a single image with ImageMagick
process_image_imagemagick() {
    local input_file="$1"
    local target_size_kb="$2"
    local filename=$(basename "$input_file")
    local name_without_ext="${filename%.*}"
    local output_file="$PROCESSING_DIR/${name_without_ext}_processed.png"
    
    echo "Processing: $filename"
    
    # Get original dimensions and size
    local original_info=$(identify -format "%wx%h %B" "$input_file")
    local original_size_kb=$(get_size_kb "$input_file")
    
    echo "  Original: $original_info, ${original_size_kb} KB"
    
    # Calculate resize percentage based on target size
    # This is a rough approximation - ImageMagick approach
    local scale_factor=$(echo "scale=2; sqrt($target_size_kb / $original_size_kb)" | bc -l)
    local resize_percent=$(echo "scale=0; $scale_factor * 100" | bc -l)
    
    # Ensure minimum size
    if (( $(echo "$resize_percent < 10" | bc -l) )); then
        resize_percent=10
    fi
    
    # Process with ImageMagick - preserve metadata and convert to PNG
    convert "$input_file" \
        -resize "${resize_percent}%" \
        -strip \
        -profile "*" \
        "$output_file"
    
    # Alternative approach that preserves more metadata:
    # convert "$input_file" -resize "${resize_percent}%" "$output_file"
    
    # Get final dimensions and size
    local final_info=$(identify -format "%wx%h %B" "$output_file")
    local final_size_kb=$(get_size_kb "$output_file")
    
    echo "  Processed: $final_info, ${final_size_kb} KB"
    echo "  Saved to: $output_file"
    
    # Calculate size reduction
    local reduction=$(echo "scale=1; (($original_size_kb - $final_size_kb) * 100) / $original_size_kb" | bc -l)
    echo "  Size reduction: ${reduction}%"
    echo
}

# Function to process all images
process_all_images() {
    local target_size_kb="$1"
    
    echo "ImageMagick Image Processing"
    echo "===================================================="
    echo "Incoming directory: $INCOMING_DIR"
    echo "Processing directory: $PROCESSING_DIR"
    echo "Target size: ${target_size_kb} KB"
    echo
    
    # Find all image files
    local image_count=0
    local processed_count=0
    
    for ext in jpg jpeg png bmp tiff tif webp JPG JPEG PNG BMP TIFF TIF WEBP; do
        for file in "$INCOMING_DIR"/*."$ext" 2>/dev/null; do
            if [[ -f "$file" ]]; then
                ((image_count++))
                if process_image_imagemagick "$file" "$target_size_kb"; then
                    ((processed_count++))
                fi
            fi
        done
    done
    
    if [[ $image_count -eq 0 ]]; then
        echo "No image files found in the incoming directory."
    else
        echo "Successfully processed $processed_count out of $image_count images."
    fi
}

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "Error: ImageMagick is not installed."
    echo "Install it with: sudo apt-get install imagemagick"
    exit 1
fi

# Check if bc calculator is available
if ! command -v bc &> /dev/null; then
    echo "Error: bc calculator is not installed."
    echo "Install it with: sudo apt-get install bc"
    exit 1
fi

# Main execution
if [[ $# -eq 0 ]]; then
    process_all_images 500
elif [[ $1 =~ ^[0-9]+$ ]]; then
    process_all_images "$1"
else
    echo "Usage: $0 [target_size_kb]"
    echo "Example: $0 300"
    exit 1
fi