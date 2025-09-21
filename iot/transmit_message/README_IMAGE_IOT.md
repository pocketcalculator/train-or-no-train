# Image Processing and IoT Hub Integration

This solution processes images from the incoming directory and sends them as telemetry data to Azure IoT Hub with EXIF timestamp information and a training flag.

## Overview

The integrated system performs these steps:
1. **Image Processing**: Takes an image from the incoming directory and creates an optimized version for IoT transmission
2. **EXIF Extraction**: Extracts the timestamp from the original image's EXIF data  
3. **IoT Transmission**: Sends the processed image, timestamp, and train=true flag to Azure IoT Hub

## Files

- `process_and_send_image.py` - Main integration script
- `run_image_to_iot.sh` - Convenience script to run the process
- `README_IMAGE_IOT.md` - This documentation

## Setup

### Prerequisites
The script requires these Python packages (already installed in iot-venv):
- `azure-iot-device` - For IoT Hub communication
- `Pillow` - For image processing
- `piexif` - For EXIF data handling

### IoT Hub Configuration
The script is already configured with the IoT Hub connection string:
```
HostName=msfthack2025IoTHub.azure-devices.net;DeviceId=railroadEdgeDevice
```

## Usage

### Method 1: Using the convenience script
```bash
# Process the first image found in incoming directory
./run_image_to_iot.sh

# Process a specific image
./run_image_to_iot.sh PXL_20250916_003238285.jpg
```

### Method 2: Direct Python execution
```bash
cd /home/kb1hgo/iot
source ../iot-venv/bin/activate

# Process first available image
python3 process_and_send_image.py

# Process specific image
python3 process_and_send_image.py PXL_20250916_003238285.jpg

# Test IoT Hub connection
python3 process_and_send_image.py --test-connection
```

## How It Works

### Step 1: Image Optimization
- Takes the original image from `/home/kb1hgo/image/incoming/`
- Creates a small version optimized for IoT Hub transmission (max 60KB target)
- Automatically reduces dimensions and applies compression
- Saves the small version to `/home/kb1hgo/image/processing/` with `_iot_small.png` suffix

### Step 2: EXIF Timestamp Extraction  
- Extracts timestamp from the original image's EXIF data
- Looks for `DateTime`, `DateTimeOriginal`, or `DateTimeDigitized` fields
- Falls back to current time if no EXIF timestamp is found
- Format: "YYYY:MM:DD HH:MM:SS"

### Step 3: IoT Hub Transmission
- Encodes the small image as base64
- Creates JSON message with:
  - `imageData`: Base64 encoded image
  - `imageTimestamp`: EXIF timestamp from original image
  - `train`: Always set to `true`
  - `deviceId`: "railroadEdgeDevice"
  - `messageTimestamp`: Current UTC timestamp
  - Additional metadata fields

### Message Size Limits
- Azure IoT Hub has a 256KB message size limit
- The script automatically creates images small enough to fit this limit
- Typical message size: 50-80KB (well under the limit)

## Example Output

```
🖼️ Processing Image and Sending to IoT Hub
============================================================
📁 Input file: /home/kb1hgo/image/incoming/PXL_20250916_003238285.jpg

🔄 Step 1: Creating IoT-optimized image...
  Final: 192x144, 40.6 KB
✅ Small image created: PXL_20250916_003238285_iot_small.png (40.6 KB)

🔄 Step 2: Extracting EXIF timestamp from original image...
✅ Found timestamp: 2025:09:15 20:32:38

🔄 Step 3: Encoding small image to base64...
✅ Image encoded (estimated message size: 72.0 KB)

🔄 Step 4: Sending to IoT Hub...
✅ Message sent successfully!

🎉 Successfully completed all steps!
📊 Summary:
   Original image: PXL_20250916_003238285.jpg
   Small image: PXL_20250916_003238285_iot_small.png
   Timestamp: 2025:09:15 20:32:38
   Train flag: True
   Sent to IoT Hub: ✅
```

## Message Structure

The JSON message sent to IoT Hub contains:

```json
{
    "deviceId": "railroadEdgeDevice",
    "messageTimestamp": "2025-09-17T15:30:45.123456",
    "imageTimestamp": "2025:09:15 20:32:38",
    "train": true,
    "imageData": "iVBORw0KGgoAAAANSUhEUgAA...[base64 image data]",
    "messageType": "imageProcessing",
    "imageFormat": "png",
    "processed": true
}
```

## Integration with Existing Scripts

This solution integrates with:
- **Image Processing**: Uses functions from `/home/kb1hgo/image/process_image.py`
- **IoT Messaging**: Based on `/home/kb1hgo/iot/send_iot_message.py`
- **Directories**: 
  - Source: `/home/kb1hgo/image/incoming/`
  - Output: `/home/kb1hgo/image/processing/`

## Monitoring Messages

You can monitor the sent messages using Azure CLI:
```bash
az iot hub monitor-events --hub-name msfthack2025IoTHub --device-id railroadEdgeDevice
```

Or use the provided monitoring script:
```bash
cd /home/kb1hgo/iot
python3 read_iot_messages.py
```

## Troubleshooting

- **"No image files found"**: Place image files in `/home/kb1hgo/image/incoming/`
- **"IoT Hub connection failed"**: Check network connectivity and IoT Hub status
- **"Message too large"**: The script automatically handles this, but very large images might need manual intervention
- **"No timestamp found"**: Script uses current time as fallback, this is normal for images without EXIF data