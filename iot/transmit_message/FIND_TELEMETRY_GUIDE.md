# Finding Telemetry Data in Azure Portal - Step-by-Step Guide

## 🎯 You Just Sent a Message Successfully!

Since you just ran `./run_iot_sender.sh` and it worked, there should be fresh telemetry data to view. Let's find it!

---

## 📍 Method 1: IoT Hub Metrics (Most Reliable)

### Step-by-Step Instructions:

1. **Open Azure Portal**
   - Go to: https://portal.azure.com
   - Sign in with your credentials

2. **Navigate to Your IoT Hub**
   - In the search bar at the top, type: `msfthack2025IoTHub`
   - Click on your IoT Hub when it appears

3. **Go to Metrics**
   - In the left sidebar, look for **"Monitoring"** section
   - Click **"Metrics"**

4. **Add Telemetry Metric**
   - Click **"+ Add metric"**
   - **Metric Namespace**: `IoT Hub standard metrics`
   - **Metric**: Select **"Telemetry messages sent"**
   - **Aggregation**: `Sum` or `Count`

5. **Set Time Range**
   - Click the time picker (usually shows "Last 24 hours")
   - Change to **"Last hour"** or **"Last 30 minutes"**
   - Click **"Apply"**

6. **Look for Your Message**
   - You should see a spike in the chart where you sent the message
   - The spike should be around the time you just ran the script

---

## 📍 Method 2: Device-Specific Telemetry

### Step-by-Step Instructions:

1. **Navigate to Devices**
   - In your IoT Hub, click **"Devices"** (under "Device management" section)
   - You should see your device: **railroadEdgeDevice**

2. **Open Device Details**
   - Click on **railroadEdgeDevice**

3. **Check Device Telemetry**
   - Look for tabs at the top: **Overview**, **Device twin**, **Direct method**, etc.
   - Click **"Device telemetry"** tab (if available)
   - Or look for **"Telemetry"** section in the Overview

4. **Alternative: Device-to-Cloud Messages**
   - If you don't see "Telemetry", look for **"Device-to-cloud messages"**
   - Or **"Message monitoring"**

---

## 📍 Method 3: Built-in Endpoints

### Step-by-Step Instructions:

1. **Go to Built-in Endpoints**
   - In your IoT Hub, click **"Built-in endpoints"** (under "Hub settings")

2. **View Events Endpoint**
   - You'll see information about the **"Events"** endpoint
   - This shows where your messages are stored

3. **Check Consumer Groups**
   - Look at the **"Consumer groups"** section
   - Your messages are available through the `$Default` consumer group

---

## 📍 Method 4: Monitoring Overview

### Alternative View:

1. **IoT Hub Overview**
   - From your IoT Hub main page
   - Scroll down to see usage charts

2. **Usage Statistics**
   - Look for **"Messages used today"**
   - **"Total devices"**
   - **"Connected devices"**

---

## 🔍 Troubleshooting: If You Still Don't See Data

### Check These Common Issues:

1. **Time Zone Issues**
   - Make sure the time range includes when you sent the message
   - Try expanding to "Last 4 hours" to be safe

2. **Metric Refresh**
   - Click the **"Refresh"** button in the metrics view
   - Sometimes there's a 2-5 minute delay

3. **Device Status**
   - Go to **Devices** → **railroadEdgeDevice**
   - Check if **"Connection state"** shows as **"Connected"** or **"Disconnected"**
   - Check **"Last activity time"**

4. **Message Count**
   - In the device details, look for **"Telemetry sent"** count
   - This should increment each time you send a message

---

## 📊 What Your Telemetry Data Looks Like

When you find it, your message should contain:

```json
{
  "deviceId": "raspberry-pi-5",
  "timestamp": "2025-09-17T[current-time]",
  "message": "Hello from Raspberry Pi 5!",
  "temperature": 23.5,
  "humidity": 65.2,
  "platform": "Linux-6.12.25+rpt-rpi-2712-aarch64-with-glibc2.36",
  "python_version": "3.11.2"
}
```

Plus Azure system properties like:
- Device ID: `railroadEdgeDevice`
- Enqueue time
- Message ID

---

## 🚀 Quick Test: Send Another Message

To verify the telemetry is updating in real-time:

1. **Keep the Azure Portal metrics page open**
2. **Run the script again**: `./run_iot_sender.sh`
3. **Wait 2-3 minutes**
4. **Click "Refresh" in the portal**
5. **You should see another spike!**

---

## 📱 Alternative: Azure Mobile App

If the web portal is giving you trouble:

1. **Download "Microsoft Azure" mobile app**
2. **Sign in with your account**
3. **Navigate to your IoT Hub**
4. **View metrics on mobile**

---

## ❓ Still Not Seeing Telemetry?

Let me know which of these methods you tried and what you're seeing:

1. **Are you seeing any metrics at all in the Metrics section?**
2. **Does the device show as "Connected" in the Devices list?**
3. **Is there a "Last activity time" showing recent activity?**
4. **Are you looking in the right time range (last hour)?**

The message was definitely sent successfully from your Raspberry Pi, so the data should be there!