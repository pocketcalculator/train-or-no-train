#!/usr/bin/env python3
"""
Simple IoT Hub metrics checker - works with existing packages
Shows message counts and suggests where to view actual message content
"""

import requests
import json
from datetime import datetime, timedelta

def show_message_viewing_options():
    """Show all the ways to view IoT Hub messages"""
    print("📨 Where to View Your IoT Hub Messages")
    print("=" * 50)
    print()
    
    print("🎯 EASIEST OPTIONS:")
    print()
    print("1. 🌐 Azure Portal - Metrics Dashboard")
    print("   • URL: https://portal.azure.com")
    print("   • Path: IoT Hubs → msfthack2025IoTHub → Monitoring → Metrics")
    print("   • Add metric: 'Telemetry messages sent'")
    print("   • Time range: Last 1 hour")
    print("   • You'll see spikes when messages are sent!")
    print()
    
    print("2. 📱 Azure Portal - Device Telemetry")
    print("   • Path: IoT Hubs → msfthack2025IoTHub → Devices")
    print("   • Click: railroadEdgeDevice")
    print("   • Tab: 'Telemetry' (shows recent messages)")
    print()
    
    print("3. 🔧 VS Code with Azure IoT Tools")
    print("   • Install extension: 'Azure IoT Tools'")
    print("   • Sign in to Azure account")
    print("   • Explorer → Azure IoT Hub → Your Hub")
    print("   • Right-click → 'Start Monitoring Built-in Event Endpoint'")
    print()
    
    print("🔧 ADVANCED OPTIONS:")
    print()
    print("4. 📊 Azure Monitor Workbooks")
    print("   • IoT Hub → Monitoring → Workbooks")
    print("   • Select 'IoT Hub Insights'")
    print()
    
    print("5. 📈 Log Analytics (if configured)")
    print("   • Set up: IoT Hub → Diagnostic settings")
    print("   • Query with: Azure Monitor → Logs")
    print()
    
    print("6. 🔔 Real-time Alerts")
    print("   • IoT Hub → Monitoring → Alerts")
    print("   • Create alerts for message thresholds")

def show_portal_steps():
    """Show detailed steps for Azure Portal"""
    print("🌐 Step-by-Step: Azure Portal")
    print("=" * 40)
    print()
    print("1. Open browser to: https://portal.azure.com")
    print("2. Sign in with your Azure credentials")
    print("3. In the search bar, type: msfthack2025IoTHub")
    print("4. Click on your IoT Hub")
    print("5. In the left menu, click: 'Monitoring' → 'Metrics'")
    print("6. Click '+ Add metric'")
    print("7. Select:")
    print("   • Metric: 'Telemetry messages sent'")
    print("   • Aggregation: 'Count'")
    print("   • Time range: 'Last hour'")
    print("8. You should see spikes where messages were sent!")
    print()
    print("💡 Pro tip: You can also check 'Connected devices' metric")
    print("   to see when your Raspberry Pi connects/disconnects")

def test_message_sending():
    """Send a test message to generate activity"""
    print("🧪 Testing Message Sending")
    print("=" * 30)
    print()
    
    try:
        import subprocess
        import sys
        
        print("📤 Sending test message to IoT Hub...")
        result = subprocess.run([
            sys.executable, 
            "/home/kb1hgo/send_iot_message.py"
        ], capture_output=True, text=True, cwd="/home/kb1hgo")
        
        if result.returncode == 0:
            print("✅ SUCCESS! Test message sent at:", datetime.now().strftime("%H:%M:%S"))
            print()
            print("🌐 Now check Azure Portal:")
            print("   1. Go to: https://portal.azure.com")
            print("   2. Navigate: IoT Hubs → msfthack2025IoTHub → Monitoring → Metrics")
            print("   3. Look for the spike in 'Telemetry messages sent'")
            print(f"   4. Time to look for: {datetime.now().strftime('%H:%M')} (just now)")
            print()
            print("📊 Your message content was:")
            # Extract the message from the output
            if "Sending message:" in result.stdout:
                message_line = [line for line in result.stdout.split('\n') if 'Sending message:' in line]
                if message_line:
                    try:
                        message_json = message_line[0].split('Sending message: ')[1]
                        parsed = json.loads(message_json)
                        print(json.dumps(parsed, indent=2))
                    except:
                        print("   (Message sent successfully but couldn't parse content)")
        else:
            print("❌ Error sending message:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running test: {e}")

def main():
    """Main function"""
    print("Azure IoT Hub Message Viewer Guide")
    print("=" * 45)
    print()
    
    print("The Azure CLI monitoring doesn't work on Raspberry Pi due to")
    print("compilation issues with the uamqp library on ARM64 architecture.")
    print()
    print("But don't worry! Here are several working alternatives:")
    print()
    
    # Show options
    show_message_viewing_options()
    print()
    
    # Ask what the user wants to do
    print("🚀 What would you like to do?")
    print("1. Send a test message and show where to view it")
    print("2. Show detailed Azure Portal steps") 
    print("3. Just show the options again")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        print()
        
        if choice == "1":
            test_message_sending()
        elif choice == "2":
            show_portal_steps()
        elif choice == "3":
            show_message_viewing_options()
        else:
            print("Invalid choice. Showing all options:")
            show_message_viewing_options()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()