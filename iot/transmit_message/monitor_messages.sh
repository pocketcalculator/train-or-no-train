#!/bin/bash
echo "Azure IoT Hub Message Monitor"
echo "============================="
echo ""
echo "This script shows different ways to monitor your IoT Hub messages"
echo ""

echo "1. Monitor messages in real-time (requires Azure CLI):"
echo "   az iot hub monitor-events --hub-name msfthack2025IoTHub --device-id railroadEdgeDevice"
echo ""

echo "2. View device telemetry (last 100 messages):"
echo "   az iot hub monitor-events --hub-name msfthack2025IoTHub --device-id railroadEdgeDevice --timeout 30"
echo ""

echo "3. Monitor all devices in the hub:"
echo "   az iot hub monitor-events --hub-name msfthack2025IoTHub"
echo ""

echo "4. Show message properties and system properties:"
echo "   az iot hub monitor-events --hub-name msfthack2025IoTHub --device-id railroadEdgeDevice --properties all"
echo ""

echo "Note: You need to install Azure CLI and login first:"
echo "   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
echo "   az login"
echo "   az extension add --name azure-iot"