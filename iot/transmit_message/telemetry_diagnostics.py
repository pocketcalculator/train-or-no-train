#!/usr/bin/env python3
"""
IoT Hub Telemetry Diagnostic Tool
Helps troubleshoot why telemetry might not be visible in Azure Portal
"""

import json
import re
from datetime import datetime, timedelta

def check_connection_string():
    """Analyze the connection string for common issues"""
    print("🔍 Checking Connection String Configuration")
    print("=" * 50)
    
    try:
        with open('/home/kb1hgo/iot/send_iot_message.py', 'r') as f:
            content = f.read()
        
        # Extract connection string
        match = re.search(r'CONNECTION_STRING\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            conn_str = match.group(1)
            print("✅ Connection string found in script")
            
            # Parse connection string components
            parts = {}
            for part in conn_str.split(';'):
                if '=' in part:
                    key, value = part.split('=', 1)
                    parts[key] = value
            
            print(f"📡 IoT Hub: {parts.get('HostName', 'NOT FOUND')}")
            print(f"🔧 Device ID: {parts.get('DeviceId', 'NOT FOUND')}")
            print(f"🔑 Has SharedAccessKey: {'✅ Yes' if 'SharedAccessKey' in parts else '❌ No'}")
            
            # Verify components
            if 'HostName' in parts and 'msfthack2025IoTHub' in parts['HostName']:
                print("✅ IoT Hub name matches expected")
            else:
                print("⚠️  IoT Hub name might not match")
                
            if 'DeviceId' in parts and parts['DeviceId'] == 'railroadEdgeDevice':
                print("✅ Device ID matches expected")
            else:
                print("⚠️  Device ID might not match expected")
                
        else:
            print("❌ Connection string not found in script")
            
    except Exception as e:
        print(f"❌ Error reading script: {e}")

def check_recent_messages():
    """Show information about recent message sending"""
    print("\n📨 Recent Message Activity")
    print("=" * 30)
    
    print("✅ Last script run was successful (you confirmed this)")
    print(f"⏰ Expected message time: {datetime.now().strftime('%H:%M:%S')} (just now)")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    print("\n🎯 What to look for in Azure Portal:")
    print("- Time range: Last 1-2 hours")
    print("- Metric: 'Telemetry messages sent'")
    print("- Expected count: +1 from previous value")

def check_common_portal_issues():
    """List common reasons telemetry might not show up"""
    print("\n🔧 Common Portal Issues and Solutions")
    print("=" * 40)
    
    issues = [
        {
            "issue": "Wrong time range selected",
            "solution": "Set time range to 'Last hour' or 'Last 4 hours'"
        },
        {
            "issue": "Looking in wrong location", 
            "solution": "Go to IoT Hub → Monitoring → Metrics (not Devices → Telemetry)"
        },
        {
            "issue": "Metric not refreshed",
            "solution": "Click 'Refresh' button or wait 2-5 minutes for data"
        },
        {
            "issue": "Wrong metric selected",
            "solution": "Use 'Telemetry messages sent' not 'Messages used'"
        },
        {
            "issue": "Portal caching",
            "solution": "Try opening portal in incognito/private browser window"
        },
        {
            "issue": "Time zone confusion",
            "solution": "Portal uses UTC time, check if your local time matches"
        }
    ]
    
    for i, item in enumerate(issues, 1):
        print(f"{i}. ❌ {item['issue']}")
        print(f"   ✅ {item['solution']}")
        print()

def show_portal_urls():
    """Show direct URLs to portal sections"""
    print("🌐 Direct Portal Links")
    print("=" * 25)
    
    base_url = "https://portal.azure.com"
    
    print("📊 Metrics (Primary location for telemetry):")
    print("   Portal → Search 'msfthack2025IoTHub' → Monitoring → Metrics")
    print()
    
    print("🔧 Device Management:")
    print("   Portal → Search 'msfthack2025IoTHub' → Devices → railroadEdgeDevice")
    print()
    
    print("📈 Overview Dashboard:")
    print("   Portal → Search 'msfthack2025IoTHub' → Overview")

def show_alternative_verification():
    """Show ways to verify messages are reaching IoT Hub"""
    print("\n🔍 Alternative Verification Methods")
    print("=" * 40)
    
    print("1. 📊 Device Statistics:")
    print("   Portal → IoT Hub → Devices → railroadEdgeDevice")
    print("   Look for: 'Last activity time' (should be recent)")
    print()
    
    print("2. 🔔 Activity Log:")
    print("   Portal → IoT Hub → Activity log")
    print("   Filter by: Last 4 hours")
    print()
    
    print("3. 📋 Resource Health:")
    print("   Portal → IoT Hub → Resource health")
    print("   Should show: 'Available'")
    print()
    
    print("4. 💳 Billing/Usage:")
    print("   Portal → IoT Hub → Overview")
    print("   Look for: 'Messages used today' counter")

def run_diagnostics():
    """Run all diagnostic checks"""
    print("Azure IoT Hub Telemetry Diagnostics")
    print("=" * 45)
    print(f"🕐 Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    check_connection_string()
    check_recent_messages()
    check_common_portal_issues()
    show_portal_urls()
    show_alternative_verification()
    
    print("\n🎯 Next Steps:")
    print("1. Open Azure Portal: https://portal.azure.com")
    print("2. Search for: msfthack2025IoTHub")
    print("3. Go to: Monitoring → Metrics")
    print("4. Add metric: 'Telemetry messages sent'")
    print("5. Set time range: 'Last hour'")
    print("6. Look for spike at current time")
    print()
    print("💡 If still no data, let me know what you see in the portal!")

if __name__ == "__main__":
    run_diagnostics()