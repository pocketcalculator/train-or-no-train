#!/bin/bash

# Script to monitor Azure Function logs in real-time
# Useful for debugging and testing

echo "=== Azure Function Log Monitor ==="
echo "This script monitors the Azure Function logs in real-time"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Check if the function is running
if ! pgrep -f "func start" > /dev/null; then
    echo -e "${YELLOW}Warning:${NC} Azure Functions runtime doesn't appear to be running"
    echo "Start it with: npm start"
    echo ""
fi

# Monitor the function logs
print_info "Starting log monitoring..."
print_info "Looking for Azure Function log files..."

# Try to find and tail the Azure Functions log files
# The exact path might vary depending on your setup
if [ -d "./logs" ]; then
    tail -f ./logs/*.log 2>/dev/null || print_info "No log files found in ./logs"
elif [ -d "../logs" ]; then
    tail -f ../logs/*.log 2>/dev/null || print_info "No log files found in ../logs"
else
    print_info "No local log directory found"
    print_info "Azure Functions logs will appear in the terminal where you ran 'npm start'"
    print_info "This monitor script is mainly useful when running in production"
fi

echo ""
print_info "Tip: You can also check logs using Azure CLI:"
echo "az functionapp logs tail --name <function-app-name> --resource-group <resource-group>"
