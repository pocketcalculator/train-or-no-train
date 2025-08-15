#!/bin/bash

# Script to upload test files to Azurite blob storage
# Requires Azure CLI to be installed and configured

set -e

echo "=== Uploading Test Files to Azurite ==="

# Configuration
STORAGE_ACCOUNT="devstoreaccount1"
CONTAINER_NAME="incoming"
CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

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

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first."
    echo "Install with: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
    exit 1
fi

# Check if test data directory exists
if [ ! -d "./test-data" ]; then
    print_error "Test data directory not found. Run './test-helper.sh test-files' first."
    exit 1
fi

# Create container if it doesn't exist
print_status "Creating container if it doesn't exist..."
az storage container create \
    --name "$CONTAINER_NAME" \
    --connection-string "$CONNECTION_STRING" \
    --public-access off \
    2>/dev/null || print_warning "Container might already exist"

# Upload all PNG files from test-data directory
print_status "Uploading PNG files to container '$CONTAINER_NAME'..."

for file in ./test-data/*.png; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        print_status "Uploading $filename..."
        
        az storage blob upload \
            --file "$file" \
            --name "$filename" \
            --container-name "$CONTAINER_NAME" \
            --connection-string "$CONNECTION_STRING" \
            --overwrite
            
        print_success "Uploaded $filename"
    fi
done

print_success "All files uploaded successfully!"

# List uploaded files for verification
print_status "Verifying uploaded files..."
az storage blob list \
    --container-name "$CONTAINER_NAME" \
    --connection-string "$CONNECTION_STRING" \
    --output table

echo ""
print_success "Upload complete! The Azure Function should detect these files on its next run."
