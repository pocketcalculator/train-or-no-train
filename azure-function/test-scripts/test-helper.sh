#!/bin/bash

# Azure Function Test Helper Script
# This script helps with testing the blob monitor function locally

set -e  # Exit on any error

echo "=== Azure Function Test Helper ==="
echo "This script provides utilities for testing the blob monitor function"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Azurite is running
check_azurite() {
    print_status "Checking if Azurite storage emulator is running..."
    
    if nc -z localhost 10000 2>/dev/null; then
        print_success "Azurite blob service is running on port 10000"
        return 0
    else
        print_warning "Azurite storage emulator is not running"
        echo "Please start Azurite with: azurite --silent --location ./azurite-data --debug ./azurite-debug.log"
        echo "Or install it with: npm install -g azurite"
        return 1
    fi
}

# Install dependencies
install_deps() {
    print_status "Installing project dependencies..."
    npm install
    print_success "Dependencies installed successfully"
}

# Build the project
build_project() {
    print_status "Building TypeScript project..."
    npm run build
    print_success "Project built successfully"
}

# Start the Azure Functions runtime
start_function() {
    print_status "Starting Azure Functions local test runner..."
    print_warning "Make sure Azurite is running before starting the function"
    print_status "Note: Using local test runner (Azure Functions Core Tools not available on ARM64)"
    npm start
}

# Upload test PNG files to the incoming container
upload_test_files() {
    print_status "Uploading test PNG files..."
    
    # Create test directory if it doesn't exist
    mkdir -p ./test-data
    
    # Create a simple test PNG file (1x1 pixel)
    # This creates a minimal valid PNG file
    printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82' > ./test-data/test-image-1.png
    
    # Create another test file with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    cp ./test-data/test-image-1.png "./test-data/test-image-${TIMESTAMP}.png"
    
    print_success "Created test PNG files in ./test-data/"
    
    # Note: You'll need to manually upload these to Azurite or use Azure Storage Explorer
    print_warning "Please upload the files in ./test-data/ to your 'incoming' container using:"
    echo "  - Azure Storage Explorer"
    echo "  - Azure CLI: az storage blob upload"
    echo "  - Or the upload script: ./upload-test-files.sh"
}

# Clean up test data
cleanup() {
    print_status "Cleaning up test data..."
    rm -rf ./test-data
    rm -rf ./azurite-data
    rm -f ./azurite-debug.log
    print_success "Cleanup completed"
}

# Show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  check-azurite    Check if Azurite storage emulator is running"
    echo "  install          Install project dependencies"
    echo "  build            Build the TypeScript project"
    echo "  start            Start the Azure Functions runtime"
    echo "  test-files       Create test PNG files for testing"
    echo "  cleanup          Clean up test data and Azurite data"
    echo "  help             Show this help message"
    echo ""
    echo "Example workflow:"
    echo "  1. $0 install"
    echo "  2. $0 build"
    echo "  3. Start Azurite: azurite --silent --location ./azurite-data"
    echo "  4. $0 test-files"
    echo "  5. Upload files to incoming container"
    echo "  6. $0 start"
}

# Main command handling
case "${1:-help}" in
    "check-azurite")
        check_azurite
        ;;
    "install")
        install_deps
        ;;
    "build")
        build_project
        ;;
    "start")
        check_azurite && start_function
        ;;
    "test-files")
        upload_test_files
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        show_help
        ;;
esac
