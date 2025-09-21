# Where Your IoT Hub Messages Go

## 📨 Your Message Journey

When your Raspberry Pi sends messages to **msfthack2025IoTHub**, here's exactly what happens:

### 1. **Message Reception**
- IoT Hub receives your JSON message
- Message gets a timestamp and sequence number
- System properties are added (device ID, connection info, etc.)

### 2. **Default Storage Location**
Your messages are stored in IoT Hub's **built-in Event Hub endpoint**:
- **Endpoint name**: `messages/events`
- **Retention**: 1-7 days (depending on your IoT Hub tier)
- **Format**: Event Hub-compatible format
- **Access**: Can be read by any Event Hub-compatible client

### 3. **Where to View Your Messages**

#### **Option A: Azure Portal (Easiest)**
1. Go to: https://portal.azure.com
2. Navigate to: **IoT Hubs** → **msfthack2025IoTHub**
3. View options:
   - **Monitoring** → **Metrics**: See message counts and throughput
   - **Devices** → **railroadEdgeDevice** → **Telemetry**: Recent messages
   - **Monitoring** → **Diagnostic settings**: Configure logging

#### **Option B: Azure Portal - Built-in Endpoints**
1. In your IoT Hub: **Built-in endpoints**
2. You'll see:
   - **Event Hub-compatible endpoint**: `sb://iothub-ns-msfthack20-xxxxx-xxxxxxxx.servicebus.windows.net/`
   - **Event Hub-compatible name**: Usually your IoT Hub name
   - **Consumer groups**: `$Default` (and any custom ones)

#### **Option C: VS Code with Azure IoT Tools**
1. Install "Azure IoT Tools" extension
2. Connect to your Azure account
3. Find your IoT Hub and device
4. Right-click → "Start Monitoring Built-in Event Endpoint"

### 4. **Your Actual Messages**

Based on what you're sending, each message contains:

```json
{
    "deviceId": "raspberry-pi-5",
    "timestamp": "2025-09-17T20:08:24.870009",
    "message": "Hello from Raspberry Pi 5!",
    "temperature": 23.5,
    "humidity": 65.2,
    "platform": "Linux-6.12.25+rpt-rpi-2712-aarch64-with-glibc2.36",
    "python_version": "3.11.2"
}
```

Plus system properties added by IoT Hub:
- `iothub-connection-device-id`: railroadEdgeDevice
- `iothub-enqueuedtime`: When IoT Hub received it
- `iothub-message-source`: telemetry
- `x-opt-sequence-number`: Message sequence
- `x-opt-offset`: Position in the event stream

### 5. **Message Routing Options**

You can configure **Message Routing** to send messages to:

#### **Storage Services**
- **Azure Blob Storage**: Long-term archival
- **Azure Data Lake**: Big data analytics

#### **Processing Services**  
- **Azure Functions**: Serverless processing
- **Azure Logic Apps**: Workflow automation
- **Azure Service Bus**: Reliable queuing

#### **Analytics Services**
- **Azure Stream Analytics**: Real-time processing
- **Azure Event Hubs**: High-throughput streaming
- **Power BI**: Real-time dashboards

#### **AI/ML Services**
- **Azure Machine Learning**: Predictive analytics
- **Azure Cognitive Services**: AI processing

### 6. **Check Your Messages Right Now**

1. **Go to Azure Portal**: https://portal.azure.com
2. **Navigate to**: IoT Hubs → msfthack2025IoTHub → Monitoring → Metrics
3. **Add metric**: "Telemetry messages sent"
4. **Time range**: Last hour
5. You should see spikes when you ran the script!

### 7. **Message Retention**

- **Default retention**: 1 day (Basic tier) or 7 days (Standard tier)
- **After retention**: Messages are automatically deleted
- **For longer storage**: Set up message routing to Azure Storage

### 8. **Costs**

- **Sending messages**: Usually free up to daily limits
- **Storage**: Minimal cost for built-in retention
- **Routing**: Additional costs if you route to other services

## 🎯 Quick Action Items

1. **View your messages now**:
   - Portal → IoT Hub → Monitoring → Metrics
   - Look for "Telemetry messages sent"

2. **Set up monitoring** (optional):
   - Portal → IoT Hub → Monitoring → Diagnostic settings
   - Enable logging to see detailed message flows

3. **Extend functionality** (optional):
   - Add message routing to Azure Storage
   - Create real-time dashboards with Power BI
   - Set up alerts for specific message patterns

Your messages are successfully reaching IoT Hub and being stored in the Event Hub-compatible endpoint! 🎉