#!/usr/bin/env python3
"""
Process Image and Send to IoT Hub

This script:
1. Processes an image from the incoming directory using the image processing functionality
2. Extracts EXIF timestamp from the processed image
3. Sends the image file, timestamp, and train=true flag to Azure IoT Hub

Usage:
    python3 process_and_send_image.py [filename]
    
If no filename is provided, it processes the first image found in the incoming directory.
"""

import os
import sys
import json
import base64
import asyncio
from pathlib import Path
from datetime import datetime
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device import Message

# Import the image processing functions
sys.path.append('/home/kb1hgo/image')
from process_image import process_single_image, get_image_metadata

# Directory paths
INCOMING_DIR = Path('/home/kb1hgo/image/incoming')
PROCESSING_DIR = Path('/home/kb1hgo/image/processing')

# IoT Hub connection string - replace with your actual connection string
CONNECTION_STRING = "HostName=YOUR_IOT_HUB.azure-devices.net;DeviceId=YOUR_DEVICE_ID;SharedAccessKey=YOUR_SHARED_ACCESS_KEY"
def find_image_in_incoming():
    """Find the first image file in the incoming directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    for ext in image_extensions:
        for image_file in INCOMING_DIR.glob(f"*{ext}"):
            return image_file
        for image_file in INCOMING_DIR.glob(f"*{ext.upper()}"):
            return image_file
    
    return None

def extract_exif_timestamp(image_path):
    """Extract the timestamp from EXIF data."""
    try:
        with Image.open(image_path) as img:
            # Handle PNG files with EXIF data stored as text
            if hasattr(img, 'text') and 'EXIF' in img.text:
                try:
                    exif_data = json.loads(img.text['EXIF'])
                    # Try different timestamp fields
                    for field in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                        if field in exif_data:
                            return exif_data[field]
                except json.JSONDecodeError:
                    pass
            
            # Handle regular EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                for tag, value in exif.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    if decoded in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                        return value
        
        return None
    except Exception as e:
        print(f"Warning: Could not extract timestamp from {image_path}: {e}")
        return None

def create_small_image(input_path, max_size_kb=60):
    """Create a very small version of the image for IoT transmission."""
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                if img.mode == 'P' and 'transparency' in img.info:
                    img = img.convert('RGBA')
                elif img.mode != 'RGBA':
                    img = img.convert('RGB')
            elif img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Start with aggressive downscaling
            original_width, original_height = img.size
            
            # Target much smaller dimensions for IoT transmission
            max_dimension = 300  # Start with 300px max dimension
            
            if original_width > original_height:
                new_width = max_dimension
                new_height = int((max_dimension * original_height) / original_width)
            else:
                new_height = max_dimension
                new_width = int((max_dimension * original_width) / original_height)
            
            # Resize with high quality
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Generate output filename
            output_filename = input_path.stem + "_iot_small.png"
            output_path = PROCESSING_DIR / output_filename
            
            # Save with high compression
            resized_img.save(output_path, "PNG", optimize=True, compress_level=9)
            
            # Check size and reduce further if needed
            file_size_kb = output_path.stat().st_size / 1024
            
            # If still too large, reduce dimensions further
            attempts = 0
            while file_size_kb > max_size_kb and attempts < 5:
                attempts += 1
                max_dimension = int(max_dimension * 0.8)  # Reduce by 20% each time
                
                if original_width > original_height:
                    new_width = max_dimension
                    new_height = int((max_dimension * original_height) / original_width)
                else:
                    new_height = max_dimension
                    new_width = int((max_dimension * original_width) / original_height)
                
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_img.save(output_path, "PNG", optimize=True, compress_level=9)
                file_size_kb = output_path.stat().st_size / 1024
                
                print(f"  Attempt {attempts}: {new_width}x{new_height}, {file_size_kb:.1f} KB")
            
            print(f"  Final: {new_width}x{new_height}, {file_size_kb:.1f} KB")
            return output_path, file_size_kb
            
    except Exception as e:
        print(f"Error creating small image: {e}")
        return None, 0

def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

async def send_to_iot_hub(image_data, timestamp, train_flag=True):
    """Send the processed image data to Azure IoT Hub."""
    try:
        # Create the IoT Hub client
        print("🔗 Connecting to Azure IoT Hub...")
        client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
        
        # Connect to the IoT Hub
        await client.connect()
        print("✅ Successfully connected to IoT Hub!")
        
        # Prepare the telemetry data
        telemetry_data = {
            "deviceId": "railroadEdgeDevice",
            "messageTimestamp": datetime.utcnow().isoformat(),
            "imageTimestamp": timestamp,
            "train": train_flag,
            "imageData": image_data,
            "messageType": "imageProcessing",
            "imageFormat": "png",
            "processed": True
        }
        
        # Convert to JSON string
        message_json = json.dumps(telemetry_data)
        
        # Create the message
        message = Message(message_json)
        
        # Add message properties
        message.message_id = f"img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        message.correlation_id = "railroad-image-telemetry"
        message.content_encoding = "utf-8"
        message.content_type = "application/json"
        
        # Add custom properties
        message.custom_properties["messageType"] = "imageProcessing"
        message.custom_properties["imageTimestamp"] = timestamp or "unknown"
        message.custom_properties["train"] = str(train_flag).lower()
        
        # Send the message
        print(f"📤 Sending image data to IoT Hub...")
        print(f"   Image timestamp: {timestamp}")
        print(f"   Train flag: {train_flag}")
        print(f"   Message size: {len(message_json):,} bytes")
        
        await client.send_message(message)
        print("✅ Message sent successfully!")
        
        # Disconnect from the IoT Hub
        await client.disconnect()
        print("🔚 Disconnected from IoT Hub")
        
        return True
        
    except Exception as e:
        print(f"❌ Error sending to IoT Hub: {str(e)}")
        return False

async def process_and_send_image(filename=None):
    """Main function to process an image and send it to IoT Hub."""
    
    # Find image to process
    if filename:
        input_path = INCOMING_DIR / filename
        if not input_path.exists():
            print(f"❌ Error: File {input_path} does not exist.")
            return False
    else:
        input_path = find_image_in_incoming()
        if not input_path:
            print("❌ No image files found in the incoming directory.")
            return False
    
    print("🖼️ Processing Image and Sending to IoT Hub")
    print("=" * 60)
    print(f"📁 Input file: {input_path}")
    print()
    
    try:
        # Step 1: Create a small image suitable for IoT transmission
        print("🔄 Step 1: Creating IoT-optimized image...")
        small_image_path, file_size_kb = create_small_image(input_path, max_size_kb=60)
        
        if not small_image_path:
            print("❌ Failed to create small image")
            return False
        
        print(f"✅ Small image created: {small_image_path} ({file_size_kb:.1f} KB)")
        print()
        
        # Step 2: Extract timestamp from original image's EXIF data  
        print("🔄 Step 2: Extracting EXIF timestamp from original image...")
        timestamp = extract_exif_timestamp(input_path)
        
        if timestamp:
            print(f"✅ Found timestamp: {timestamp}")
        else:
            print("⚠️ No timestamp found in EXIF data, using current time")
            timestamp = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        print()
        
        # Step 3: Encode small image to base64
        print("🔄 Step 3: Encoding small image to base64...")
        image_base64 = encode_image_to_base64(small_image_path)
        
        if not image_base64:
            print("❌ Failed to encode image to base64")
            return False
        
        # Calculate message size (base64 encoding increases size by ~33%)
        estimated_message_size_kb = len(image_base64) * 1.33 / 1024
        print(f"✅ Image encoded (estimated message size: {estimated_message_size_kb:.1f} KB)")
        
        if estimated_message_size_kb > 250:  # Leave some margin for other data
            print(f"⚠️ Warning: Message might be too large ({estimated_message_size_kb:.1f} KB > 250 KB limit)")
        print()
        
        # Step 4: Send to IoT Hub
        print("🔄 Step 4: Sending to IoT Hub...")
        success = await send_to_iot_hub(
            image_data=image_base64,
            timestamp=timestamp,
            train_flag=True
        )
        
        if success:
            print()
            print("🎉 Successfully completed all steps!")
            print("📊 Summary:")
            print(f"   Original image: {input_path.name}")
            print(f"   Small image: {small_image_path.name}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Train flag: True")
            print(f"   Sent to IoT Hub: ✅")
            return True
        else:
            print("❌ Failed to send to IoT Hub")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process image and send to IoT Hub")
    parser.add_argument('filename', nargs='?', help='Specific file to process (optional)')
    parser.add_argument('--test-connection', action='store_true', help='Test IoT Hub connection only')
    
    args = parser.parse_args()
    
    if args.test_connection:
        async def test_connection():
            try:
                client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
                await client.connect()
                print("✅ IoT Hub connection test successful!")
                await client.disconnect()
                return True
            except Exception as e:
                print(f"❌ IoT Hub connection test failed: {e}")
                return False
        
        asyncio.run(test_connection())
        return
    
    # Check if connection string is configured
    if CONNECTION_STRING.startswith("HostName=") and "SharedAccessKey=" in CONNECTION_STRING:
        print("✅ IoT Hub connection string configured")
    else:
        print("❌ Please update the CONNECTION_STRING variable with your actual device connection string")
        return
    
    # Run the main processing
    try:
        success = asyncio.run(process_and_send_image(args.filename))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(1)

if __name__ == "__main__":
    main()