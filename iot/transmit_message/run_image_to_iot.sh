#!/bin/bash
"""
Convenience script to run the image processing and IoT Hub sender

This script:
1. Activates the IoT virtual environment
2. Runs the process_and_send_image.py script
3. Handles the image processing and IoT Hub messaging

Usage:
    ./run_image_to_iot.sh [filename]
    
If no filename is provided, it processes the first image found in the incoming directory.
"""

# Change to the IoT directory
cd /home/kb1hgo/iot

# Activate the virtual environment
source ../iot-venv/bin/activate

# Run the integrated script
if [ $# -eq 0 ]; then
    echo "🔄 Processing first image from incoming directory..."
    python3 process_and_send_image.py
else
    echo "🔄 Processing specific image: $1"
    python3 process_and_send_image.py "$1"
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Operation completed successfully!"
else
    echo ""
    echo "❌ Operation failed. Check the output above for details."
    exit 1
fi