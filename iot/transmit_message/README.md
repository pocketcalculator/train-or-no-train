# Azure IoT Hub Message Sender for Raspberry Pi 5

This setup allows you to send a simple one-time message from your Raspberry Pi 5 to Azure IoT Hub.

## Files Created

- `send_iot_message.py` - Main Python script that sends the message
- `run_iot_sender.sh` - Setup and execution script
- `README.md` - This documentation

## Setup Instructions

### 1. Get Your Device Connection String

1. Go to the [Azure Portal](https://portal.azure.com)
2. Navigate to your IoT Hub
3. Go to **Devices** (under "Device management")
4. Select your device (or create a new one if needed)
5. Copy the **Primary connection string**

### 2. Configure the Script

Edit the `send_iot_message.py` file and replace this line:
```python
CONNECTION_STRING = "YOUR_DEVICE_CONNECTION_STRING_HERE"
```

With your actual connection string:
```python
CONNECTION_STRING = "HostName=your-iot-hub.azure-devices.net;DeviceId=your-device;SharedAccessKey=your-key"
```

### 3. Run the Message Sender

Execute the setup script:
```bash
./run_iot_sender.sh
```

Or run directly with Python:
```bash
source ~/iot-venv/bin/activate
python3 send_iot_message.py
```

## What the Message Contains

The script sends a JSON message with:
- Device identification
- Current timestamp
- A simple "Hello from Raspberry Pi 5!" message
- Example sensor data (temperature, humidity)
- System information (platform, Python version)

## Example Message

```json
{
    "deviceId": "raspberry-pi-5",
    "timestamp": "2025-09-17T12:30:45.123456",
    "message": "Hello from Raspberry Pi 5!",
    "temperature": 23.5,
    "humidity": 65.2,
    "platform": "Linux-6.1.21-v8+-aarch64-with-glibc2.36",
    "python_version": "3.11.2"
}
```

## Troubleshooting

- **Connection Error**: Verify your connection string is correct
- **Device Not Found**: Make sure the device is registered in your IoT Hub
- **Network Issues**: Check your internet connection and firewall settings
- **Permission Issues**: Make sure the script has execute permissions (`chmod +x run_iot_sender.sh`)

## Next Steps

After successfully sending a message, you can:
1. View the message in Azure IoT Hub monitoring
2. Set up message routing to other Azure services
3. Extend the script to send real sensor data
4. Set up scheduled messages using cron jobs