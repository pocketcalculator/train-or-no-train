#!/usr/bin/env python3
"""
Simple script to send a one-time message from Raspberry Pi to Azure IoT Hub
"""

import asyncio
import json
import os
import platform
from datetime import datetime
from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device import Message
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# IoT Hub connection string from environment variable
CONNECTION_STRING = os.getenv('AZURE_IOT_CONNECTION_STRING')

if not CONNECTION_STRING:
    raise ValueError("AZURE_IOT_CONNECTION_STRING environment variable is required. Please set it in your .env file.")
async def send_telemetry_message():
    """
    Send a single telemetry message to Azure IoT Hub
    """
    try:
        # Create the IoT Hub client
        print("Connecting to Azure IoT Hub...")
        client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
        
        # Connect to the IoT Hub
        await client.connect()
        print("Successfully connected to IoT Hub!")
        
        # Prepare the telemetry data
        telemetry_data = {
            "deviceId": "raspberry-pi-5",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Hello from Raspberry Pi 5!",
            "temperature": 23.5,  # Example sensor data
            "humidity": 65.2,     # Example sensor data
            "platform": platform.platform(),
            "python_version": platform.python_version()
        }
        
        # Convert to JSON string
        message_json = json.dumps(telemetry_data)
        
        # Create the message
        message = Message(message_json)
        
        # Add message properties (optional)
        message.message_id = f"msg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        message.correlation_id = "raspberry-pi-telemetry"
        message.content_encoding = "utf-8"
        message.content_type = "application/json"
        
        # Send the message
        print(f"Sending message: {message_json}")
        await client.send_message(message)
        print("Message sent successfully!")
        
        # Disconnect from the IoT Hub
        await client.disconnect()
        print("Disconnected from IoT Hub")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Make sure your connection string is correct and the device is registered in IoT Hub")

def main():
    """
    Main function to run the async telemetry sender
    """
    # Check if connection string has been updated
    if CONNECTION_STRING.startswith("HostName=") and "SharedAccessKey=" in CONNECTION_STRING:
        print("✅ Connection string configured successfully!")
    else:
        print("ERROR: Please update the CONNECTION_STRING variable with your actual device connection string")
        print("\nTo get your connection string:")
        print("1. Go to Azure Portal > IoT Hub > Devices")
        print("2. Select your device")
        print("3. Copy the 'Primary connection string'")
        print("4. Update the CONNECTION_STRING variable in this script")
        return
    
    print("Starting Azure IoT Hub message sender...")
    print("Device: Raspberry Pi 5")
    print("=" * 50)
    
    # Run the async function
    asyncio.run(send_telemetry_message())

if __name__ == "__main__":
    main()
