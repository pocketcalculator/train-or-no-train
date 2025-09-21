#!/bin/bash

echo "Azure IoT Hub Message Sender Setup"
echo "=================================="

# Activate the IoT environment
source ~/iot-venv/bin/activate

echo "IoT environment activated"
echo ""

# Check if the connection string has been configured
if grep -q "HostName=" send_iot_message.py && grep -q "SharedAccessKey=" send_iot_message.py; then
    echo "✅ Connection string configured. Running the message sender..."
    echo ""
    python3 send_iot_message.py
else
    echo "⚠️  SETUP REQUIRED:"
    echo "   Please edit send_iot_message.py and update the CONNECTION_STRING variable"
    echo "   with your actual Azure IoT Hub device connection string."
    echo ""
    echo "   To get your connection string:"
    echo "   1. Go to Azure Portal > IoT Hub > Devices"
    echo "   2. Select your device (or create a new one)"
    echo "   3. Copy the 'Primary connection string'"
    echo ""
    echo "   Then run: python3 send_iot_message.py"
fi