# Azure Blob Monitor Function

An Azure Function that monitors a blob storage container for PNG files and automatically moves them to an archive container.

## Features

- **Automatic PNG Detection**: Monitors the `incoming` container for PNG files every 30 seconds
- **File Archiving**: Moves detected PNG files to the `archive` container
- **Comprehensive Logging**: Detailed console logging for troubleshooting and monitoring
- **Error Handling**: Robust error handling with detailed error messages
- **Container Auto-Creation**: Automatically creates containers if they don't exist
- **Configurable**: Easy configuration through environment variables

## Prerequisites

- Node.js 20+ (latest supported by Azure Functions)
- Azure Functions Core Tools v4
- Azure Storage Account (or Azurite for local development)

## Quick Start

1. **Install dependencies**:
   ```bash
   ./test-scripts/test-helper.sh install
   ```

2. **Build the project**:
   ```bash
   ./test-scripts/test-helper.sh build
   ```

3. **Start Azurite (for local development)**:
   ```bash
   # Install Azurite globally if not already installed
   npm install -g azurite
   
   # Start Azurite
   azurite --silent --location ./azurite-data --debug ./azurite-debug.log
   ```

4. **Create test files**:
   ```bash
   ./test-scripts/test-helper.sh test-files
   ```

5. **Upload test files** (in a new terminal):
   ```bash
   ./test-scripts/upload-test-files.sh
   ```

6. **Start the Azure Function**:
   ```bash
   ./test-scripts/test-helper.sh start
   ```

## Configuration

The function uses the following environment variables (configured in `local.settings.json`):

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Storage connection string | `UseDevelopmentStorage=true` |
| `INCOMING_CONTAINER_NAME` | Name of the incoming container | `incoming` |
| `ARCHIVE_CONTAINER_NAME` | Name of the archive container | `archive` |

## Function Behavior

1. **Timer Trigger**: Runs every 30 seconds (configurable in the code)
2. **Container Check**: Ensures both incoming and archive containers exist
3. **PNG Detection**: Scans the incoming container for files ending in `.png` (case-insensitive)
4. **Logging**: For each PNG file found, outputs: `<filename> found at <timestamp>`
5. **File Movement**: Copies the file to the archive container, then deletes from incoming
6. **Error Handling**: Continues processing other files even if one fails

## Testing Scripts

The project includes several helper scripts in the `test-scripts/` directory:

### `test-helper.sh`
Main testing utility with the following commands:
- `install` - Install project dependencies
- `build` - Build the TypeScript project
- `start` - Start the Azure Functions runtime
- `test-files` - Create test PNG files
- `check-azurite` - Check if Azurite is running
- `cleanup` - Clean up test data

### `upload-test-files.sh`
Uploads test PNG files to the Azurite storage emulator using Azure CLI.

### `monitor-logs.sh`
Monitors Azure Function logs in real-time for debugging.

## Development Workflow

1. **Setup**: Run `./test-scripts/test-helper.sh install` and `./test-scripts/test-helper.sh build`
2. **Local Storage**: Start Azurite for local blob storage emulation
3. **Test Data**: Create and upload test PNG files
4. **Run Function**: Start the Azure Function and monitor logs
5. **Verify**: Check that files are moved from incoming to archive container

## Troubleshooting

### Common Issues

1. **Function not detecting files**:
   - Check that Azurite is running on port 10000
   - Verify files are uploaded to the correct container name
   - Check the Azure Function logs for error messages

2. **Container access errors**:
   - Ensure the storage connection string is correct
   - Verify container names in environment variables

3. **TypeScript compilation errors**:
   - Run `npm install` to ensure all dependencies are installed
   - Check that Node.js version is 20 or higher

### Log Analysis

The function provides detailed logging with timestamps and operation context:
- `[STARTUP]` - Function initialization messages
- `[CONFIG]` - Configuration values
- `[timestamp]` - Runtime operation logs
- `ERROR` - Error messages with stack traces

## Production Deployment

For production deployment:

1. Update `local.settings.json` with your actual Azure Storage connection string
2. Deploy using Azure Functions Core Tools:
   ```bash
   func azure functionapp publish <your-function-app-name>
   ```
3. Configure application settings in the Azure portal or via Azure CLI

## Timer Schedule

The function runs on a timer trigger with the schedule `0 */30 * * * *` (every 30 seconds).

To modify the schedule, update the `schedule` property in the `app.timer()` call:
- Every minute: `0 */1 * * * *`
- Every 5 minutes: `0 */5 * * * *`
- Every hour: `0 0 */1 * * *`

## File Structure

```
azure-function/
├── src/
│   └── functions/
│       └── blobMonitor.ts          # Main function code
├── test-scripts/
│   ├── test-helper.sh              # Main testing utility
│   ├── upload-test-files.sh        # Upload test files
│   └── monitor-logs.sh             # Log monitoring
├── package.json                    # Node.js dependencies
├── tsconfig.json                   # TypeScript configuration
├── host.json                       # Azure Functions host config
├── local.settings.json             # Local environment variables
└── README.md                       # This file
```

## License

This project is provided as-is for demonstration purposes.
