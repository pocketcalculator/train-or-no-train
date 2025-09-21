#!/usr/bin/env python3
"""
Simple script to read messages from Azure IoT Hub
This demonstrates how to consume the messages your Raspberry Pi sends
"""

import asyncio
from azure.eventhub.aio import EventHubConsumerClient
from azure.eventhub.extensions.checkpointstoreblobaio import BlobCheckpointStore

# IoT Hub connection details
# You'll need the Event Hub-compatible connection string from IoT Hub
EVENTHUB_CONNECTION_STRING = "YOUR_EVENTHUB_CONNECTION_STRING_HERE"
EVENTHUB_NAME = "YOUR_EVENTHUB_NAME_HERE"  # Usually the IoT Hub name

# Consumer group (use $Default for basic scenarios)
CONSUMER_GROUP = "$Default"

async def on_event(partition_context, event):
    """
    Process incoming messages from IoT devices
    """
    print("\n" + "="*60)
    print(f"📨 New message received!")
    print(f"Device: {event.system_properties.get('iothub-connection-device-id', 'Unknown')}")
    print(f"Timestamp: {event.system_properties.get('iothub-enqueuedtime', 'Unknown')}")
    print(f"Partition: {partition_context.partition_id}")
    print(f"Offset: {event.offset}")
    print(f"Sequence: {event.sequence_number}")
    print("-" * 40)
    print("📄 Message Content:")
    try:
        message_body = event.body_as_str()
        print(message_body)
    except Exception as e:
        print(f"Could not decode message: {e}")
        print(f"Raw body: {event.body}")
    
    print("-" * 40)
    print("🏷️ System Properties:")
    for key, value in event.system_properties.items():
        print(f"  {key}: {value}")
    
    if event.properties:
        print("📋 Application Properties:")
        for key, value in event.properties.items():
            print(f"  {key}: {value}")
    
    print("="*60)
    
    # Update checkpoint so we don't re-read this message
    await partition_context.update_checkpoint(event)

async def main():
    """
    Main function to start consuming messages
    """
    # Check if connection details are configured
    if EVENTHUB_CONNECTION_STRING == "YOUR_EVENTHUB_CONNECTION_STRING_HERE":
        print("❌ Setup required!")
        print("\nTo read messages from IoT Hub, you need:")
        print("1. Event Hub-compatible connection string")
        print("2. Event Hub-compatible name")
        print("\nTo get these:")
        print("1. Go to Azure Portal > IoT Hub > Built-in endpoints")
        print("2. Copy 'Event Hub-compatible endpoint' and 'Event Hub-compatible name'")
        print("3. Update this script with those values")
        print("\nAlternatively, use Azure CLI:")
        print("   az iot hub monitor-events --hub-name msfthack2025IoTHub --device-id railroadEdgeDevice")
        return
    
    try:
        print("🔗 Connecting to IoT Hub Event Hub endpoint...")
        print(f"📡 Listening for messages from IoT devices...")
        print("💡 Press Ctrl+C to stop\n")
        
        # Create the consumer client
        client = EventHubConsumerClient.from_connection_string(
            EVENTHUB_CONNECTION_STRING,
            consumer_group=CONSUMER_GROUP,
            eventhub_name=EVENTHUB_NAME
        )
        
        # Start consuming messages
        async with client:
            await client.receive(
                on_event=on_event,
                starting_position="-1"  # Start from the latest message
            )
            
    except KeyboardInterrupt:
        print("\n👋 Stopped monitoring messages")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Check your connection string")
        print("- Verify the Event Hub name")
        print("- Ensure your IoT Hub is accessible")

if __name__ == "__main__":
    asyncio.run(main())