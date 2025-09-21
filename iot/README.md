# IoT Components Setup

This directory contains IoT components for image processing and transmission to Azure IoT Hub.

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Edit the `.env` file with your actual Azure IoT Hub connection string:

```bash
# .env file
AZURE_IOT_CONNECTION_STRING="HostName=YOUR_IOT_HUB.azure-devices.net;DeviceId=YOUR_DEVICE_ID;SharedAccessKey=YOUR_SHARED_ACCESS_KEY"
IMAGE_QUALITY=85
MAX_IMAGE_SIZE=1024
```

### 3. Get Your Azure IoT Connection String

1. Go to your Azure IoT Hub in the Azure Portal
2. Navigate to **Shared access policies** or **Device management > Devices**
3. Select your device or create a new one
4. Copy the **Primary connection string**

### 4. Usage

For message transmission:
```bash
cd transmit_message
cp .env.example .env  # Edit with your values
python3 send_iot_message.py
```

For image processing and transmission:
```bash
cd transmit_message  
python3 process_and_send_image.py
```

## Security Best Practices

- ✅ **Never commit .env files** - they contain secrets
- ✅ **Use environment variables** for all sensitive data
- ✅ **Keep .env.example updated** as a template
- ✅ **Rotate connection strings** regularly
- ✅ **Use different .env files** for different environments (dev, staging, prod)

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_IOT_CONNECTION_STRING` | Azure IoT Hub device connection string | **Required** |
| `IMAGE_QUALITY` | JPEG compression quality (1-100) | 85 |
| `MAX_IMAGE_SIZE` | Maximum image dimension in pixels | 1024 |
| `MESSAGE_INTERVAL_SECONDS` | Seconds between telemetry messages | 30 |
| `ENABLE_TELEMETRY` | Enable telemetry transmission | true |
| `DEBUG_MODE` | Enable debug logging | false |

## Troubleshooting

### Connection Issues
- Verify your connection string is correct
- Check that your device is registered in Azure IoT Hub
- Ensure your device is not disabled in the portal

### Import Errors
- Install dependencies: `pip install -r requirements.txt`
- Activate virtual environment if using one

### Environment Variable Issues
- Ensure `.env` file exists in the correct directory
- Check that variables are properly formatted (no spaces around =)
- Verify the `.env` file is in the same directory as your Python script