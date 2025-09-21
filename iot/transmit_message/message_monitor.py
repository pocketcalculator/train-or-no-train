#!/usr/bin/env python3
"""
Simple IoT Hub message monitor using Azure Event Hubs SDK
This works around the Azure CLI IoT extension issues on Raspberry Pi
"""

import asyncio
import json
import sys
from datetime import datetime

try:
    from azure.eventhub.aio import EventHubConsumerClient
    EVENTHUB_AVAILABLE = True
except ImportError:
    EVENTHUB_AVAILABLE = False

# IoT Hub built-in endpoint connection details
# You need to get these from Azure Portal > IoT Hub > Built-in endpoints
EVENTHUB_CONNECTION_STRING = "YOUR_EVENTHUB_CONNECTION_STRING_HERE"
EVENTHUB_NAME = "YOUR_EVENTHUB_NAME_HERE"
CONSUMER_GROUP = "$Default"

def show_setup_instructions():
    """Show how to get the connection details"""
    print("🔧 Setup Required")
    print("=" * 50)
    print()
    print("To monitor messages, you need the Event Hub-compatible connection details:")
    print()
    print("1. Go to Azure Portal: https://portal.azure.com")
    print("2. Navigate to: IoT Hubs → msfthack2025IoTHub")
    print("3. Click: 'Built-in endpoints' (under Settings)")
    print("4. Copy these values:")
    print("   - Event Hub-compatible endpoint")
    print("   - Event Hub-compatible name")
    print("   - Shared access policy key")
    print()
    print("5. Update this script with those values")
    print()
    print("📋 Example connection string format:")
    print("Endpoint=sb://iothub-ns-xxx.servicebus.windows.net/;SharedAccessKeyName=iothubowner;SharedAccessKey=xxx")
    print()
    print("💡 Alternative: Use Azure Portal to view messages:")
    print("   IoT Hub → Monitoring → Metrics → 'Telemetry messages sent'")

async def monitor_messages():
    """Monitor messages from IoT Hub"""
    if not EVENTHUB_AVAILABLE:
        print("❌ Azure Event Hubs SDK not available")
        print("Install with: pip install azure-eventhub")
        return

    if EVENTHUB_CONNECTION_STRING == "YOUR_EVENTHUB_CONNECTION_STRING_HERE":
        show_setup_instructions()
        return

    try:
        print("🔗 Connecting to IoT Hub Event Hub endpoint...")
        print(f"📡 Hub: msfthack2025IoTHub")
        print(f"🔍 Looking for messages from device: railroadEdgeDevice")
        print("💡 Press Ctrl+C to stop monitoring")
        print("=" * 60)

        # Create the consumer client
        client = EventHubConsumerClient.from_connection_string(
            EVENTHUB_CONNECTION_STRING,
            consumer_group=CONSUMER_GROUP,
            eventhub_name=EVENTHUB_NAME
        )

        async def on_event(partition_context, event):
            """Process incoming IoT messages"""
            device_id = event.system_properties.get('iothub-connection-device-id', 'Unknown')
            enqueued_time = event.system_properties.get('iothub-enqueuedtime', datetime.now())
            
            print(f"\n📨 Message from device: {device_id}")
            print(f"⏰ Time: {enqueued_time}")
            print(f"📦 Partition: {partition_context.partition_id}")
            print("-" * 40)
            
            try:
                message_body = event.body_as_str()
                # Try to parse as JSON for pretty printing
                try:
                    message_json = json.loads(message_body)
                    print("📄 Message content:")
                    print(json.dumps(message_json, indent=2))
                except json.JSONDecodeError:
                    print(f"📄 Message content: {message_body}")
            except Exception as e:
                print(f"❌ Could not decode message: {e}")
            
            print("=" * 60)
            await partition_context.update_checkpoint(event)

        # Start consuming
        async with client:
            await client.receive(
                on_event=on_event,
                starting_position="-1"  # Start from latest
            )

    except KeyboardInterrupt:
        print("\n👋 Stopped monitoring messages")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("- Verify your Event Hub connection string")
        print("- Check the Event Hub name")
        print("- Ensure IoT Hub is accessible")

def show_alternatives():
    """Show alternative ways to view messages"""
    print("📱 Alternative Ways to View Your Messages")
    print("=" * 50)
    print()
    print("1. 🌐 Azure Portal (Recommended):")
    print("   • Go to: https://portal.azure.com")
    print("   • Navigate: IoT Hubs → msfthack2025IoTHub")
    print("   • View: Monitoring → Metrics")
    print("   • Add metric: 'Telemetry messages sent'")
    print()
    print("2. 📊 Device Telemetry in Portal:")
    print("   • IoT Hub → Devices → railroadEdgeDevice")
    print("   • Click the 'Telemetry' tab")
    print()
    print("3. 🔧 VS Code Extension:")
    print("   • Install 'Azure IoT Tools' extension")
    print("   • Sign in to Azure")
    print("   • Right-click IoT Hub → 'Start Monitoring Built-in Event Endpoint'")
    print()
    print("4. 📈 Azure Monitor Logs:")
    print("   • Set up diagnostic settings in IoT Hub")
    print("   • Query logs with KQL (Kusto Query Language)")
    print()
    print("5. ⚡ Send test message and check portal:")

def main():
    """Main function"""
    print("Azure IoT Hub Message Monitor")
    print("=" * 40)
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_alternatives()
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "--alternatives":
        show_alternatives()
        return
    
    if EVENTHUB_AVAILABLE and EVENTHUB_CONNECTION_STRING != "YOUR_EVENTHUB_CONNECTION_STRING_HERE":
        print("▶️  Starting message monitoring...")
        asyncio.run(monitor_messages())
    else:
        if not EVENTHUB_AVAILABLE:
            print("⚠️  Azure Event Hubs SDK not installed")
            print("   To install: pip install azure-eventhub")
            print()
        
        print("📋 Easy alternatives to view your messages:")
        print()
        print("1. 🌐 Azure Portal (Easiest):")
        print("   https://portal.azure.com → IoT Hubs → msfthack2025IoTHub → Monitoring")
        print()
        print("2. 🧪 Send a test message and check metrics:")
        
        # Option to send a test message
        response = input("\n🚀 Send a test message now? (y/n): ").strip().lower()
        if response == 'y':
            import subprocess
            try:
                print("📤 Sending test message...")
                result = subprocess.run([
                    sys.executable, 
                    "/home/kb1hgo/send_iot_message.py"
                ], capture_output=True, text=True, cwd="/home/kb1hgo")
                
                if result.returncode == 0:
                    print("✅ Test message sent successfully!")
                    print("🌐 Now check Azure Portal:")
                    print("   Portal → IoT Hub → Monitoring → Metrics → 'Telemetry messages sent'")
                else:
                    print("❌ Error sending message:")
                    print(result.stderr)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print()
        print("💡 For more options, run: python3 message_monitor.py --alternatives")

if __name__ == "__main__":
    main()