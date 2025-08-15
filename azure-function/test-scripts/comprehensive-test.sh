#!/bin/bash

# Comprehensive testing script for the Azure Blob Monitor Function
# Creates various test scenarios to validate function behavior

set -e

echo "=== Comprehensive Azure Function Testing ==="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Configuration
TEST_DATA_DIR="./test-data"
STORAGE_ACCOUNT="devstoreaccount1"
CONTAINER_NAME="incoming"
ARCHIVE_CONTAINER="archive"
CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"

# Create comprehensive test data
create_comprehensive_test_data() {
    print_status "Creating comprehensive test data..."
    
    mkdir -p "$TEST_DATA_DIR"
    
    # Create multiple PNG files with different naming patterns
    create_test_png "simple.png"
    create_test_png "with-spaces in name.png"
    create_test_png "with_underscore.png"
    create_test_png "with-hyphens.png"
    create_test_png "UPPERCASE.PNG"
    create_test_png "MixedCase.Png"
    create_test_png "numbers123.png"
    create_test_png "$(date +%Y%m%d_%H%M%S)_timestamp.png"
    
    # Create non-PNG files to ensure they're ignored
    echo "This is a text file" > "$TEST_DATA_DIR/textfile.txt"
    echo "This is not a PNG" > "$TEST_DATA_DIR/fakepng.jpg"
    echo "Another non-PNG" > "$TEST_DATA_DIR/document.pdf"
    
    # Create a file with .png extension but not actually a PNG (edge case)
    echo "This is not really a PNG" > "$TEST_DATA_DIR/fake.png"
    
    print_success "Created comprehensive test data in $TEST_DATA_DIR"
    ls -la "$TEST_DATA_DIR"
}

# Create a minimal valid PNG file
create_test_png() {
    local filename="$1"
    local filepath="$TEST_DATA_DIR/$filename"
    
    # Create a minimal 1x1 pixel PNG file
    printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82' > "$filepath"
    
    print_status "Created PNG file: $filename"
}

# Setup test containers
setup_containers() {
    print_status "Setting up test containers..."
    
    # Create incoming container
    az storage container create \
        --name "$CONTAINER_NAME" \
        --connection-string "$CONNECTION_STRING" \
        --public-access off \
        2>/dev/null || print_warning "Incoming container might already exist"
    
    # Create archive container
    az storage container create \
        --name "$ARCHIVE_CONTAINER" \
        --connection-string "$CONNECTION_STRING" \
        --public-access off \
        2>/dev/null || print_warning "Archive container might already exist"
    
    print_success "Containers setup completed"
}

# Upload all test files
upload_all_test_files() {
    print_status "Uploading all test files to incoming container..."
    
    for file in "$TEST_DATA_DIR"/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            print_status "Uploading: $filename"
            
            az storage blob upload \
                --file "$file" \
                --name "$filename" \
                --container-name "$CONTAINER_NAME" \
                --connection-string "$CONNECTION_STRING" \
                --overwrite \
                --no-progress
        fi
    done
    
    print_success "All test files uploaded"
}

# List files in containers
list_container_contents() {
    local container="$1"
    print_status "Contents of '$container' container:"
    
    az storage blob list \
        --container-name "$container" \
        --connection-string "$CONNECTION_STRING" \
        --query "[].{Name:name, Size:properties.contentLength, LastModified:properties.lastModified}" \
        --output table 2>/dev/null || print_warning "Container '$container' might be empty or not exist"
}

# Monitor function execution
monitor_function() {
    print_status "Monitoring function execution..."
    print_warning "The function runs every 30 seconds. Monitoring for 2 minutes..."
    
    # Show initial state
    echo ""
    print_status "=== BEFORE FUNCTION EXECUTION ==="
    list_container_contents "$CONTAINER_NAME"
    echo ""
    list_container_contents "$ARCHIVE_CONTAINER"
    echo ""
    
    # Wait for function to process files (2 execution cycles)
    print_status "Waiting for function to process files (120 seconds)..."
    for i in {1..24}; do
        echo -n "."
        sleep 5
    done
    echo ""
    
    # Show final state
    print_status "=== AFTER FUNCTION EXECUTION ==="
    list_container_contents "$CONTAINER_NAME"
    echo ""
    list_container_contents "$ARCHIVE_CONTAINER"
    echo ""
}

# Validate results
validate_results() {
    print_status "Validating test results..."
    
    # Count PNG files that should have been moved
    local expected_png_count=8  # We created 8 PNG files
    
    # Count files in archive
    local archive_count=$(az storage blob list \
        --container-name "$ARCHIVE_CONTAINER" \
        --connection-string "$CONNECTION_STRING" \
        --query "length([?ends_with(name, '.png') || ends_with(name, '.PNG') || ends_with(name, '.Png')])" \
        --output tsv 2>/dev/null || echo "0")
    
    # Count remaining PNG files in incoming
    local remaining_png_count=$(az storage blob list \
        --container-name "$CONTAINER_NAME" \
        --connection-string "$CONNECTION_STRING" \
        --query "length([?ends_with(name, '.png') || ends_with(name, '.PNG') || ends_with(name, '.Png')])" \
        --output tsv 2>/dev/null || echo "0")
    
    # Count non-PNG files that should remain in incoming
    local remaining_non_png_count=$(az storage blob list \
        --container-name "$CONTAINER_NAME" \
        --connection-string "$CONNECTION_STRING" \
        --query "length([?!ends_with(name, '.png') && !ends_with(name, '.PNG') && !ends_with(name, '.Png')])" \
        --output tsv 2>/dev/null || echo "0")
    
    echo ""
    print_status "=== TEST RESULTS ==="
    echo "Expected PNG files to move: $expected_png_count"
    echo "PNG files found in archive: $archive_count"
    echo "PNG files remaining in incoming: $remaining_png_count"
    echo "Non-PNG files remaining in incoming: $remaining_non_png_count"
    echo ""
    
    # Validate results
    if [ "$archive_count" -eq "$expected_png_count" ] && [ "$remaining_png_count" -eq 0 ]; then
        print_success "✅ TEST PASSED: All PNG files were correctly moved to archive"
    else
        print_warning "⚠️  TEST RESULTS: Some PNG files may not have been processed yet"
        print_warning "This could be normal if the function hasn't run yet or encountered errors"
    fi
    
    if [ "$remaining_non_png_count" -gt 0 ]; then
        print_success "✅ NON-PNG FILES: Correctly ignored non-PNG files"
    else
        print_warning "⚠️  NON-PNG FILES: Expected some non-PNG files to remain"
    fi
}

# Cleanup test data
cleanup_test_data() {
    print_status "Cleaning up test data..."
    
    # Clean local test data
    rm -rf "$TEST_DATA_DIR"
    
    # Clean containers (optional)
    read -p "Do you want to clean up the blob containers? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up containers..."
        
        # Delete all blobs in incoming container
        az storage blob delete-batch \
            --source "$CONTAINER_NAME" \
            --connection-string "$CONNECTION_STRING" \
            2>/dev/null || print_warning "Could not clean incoming container"
        
        # Delete all blobs in archive container
        az storage blob delete-batch \
            --source "$ARCHIVE_CONTAINER" \
            --connection-string "$CONNECTION_STRING" \
            2>/dev/null || print_warning "Could not clean archive container"
        
        print_success "Containers cleaned"
    fi
    
    print_success "Cleanup completed"
}

# Run performance test
run_performance_test() {
    print_status "Running performance test with multiple files..."
    
    mkdir -p "$TEST_DATA_DIR/perf"
    
    # Create 20 PNG files for performance testing
    for i in {1..20}; do
        create_test_png "perf/perf_test_$i.png"
    done
    
    print_status "Created 20 PNG files for performance testing"
    
    # Upload performance test files
    for file in "$TEST_DATA_DIR/perf"/*.png; do
        if [ -f "$file" ]; then
            filename="perf_$(basename "$file")"
            az storage blob upload \
                --file "$file" \
                --name "$filename" \
                --container-name "$CONTAINER_NAME" \
                --connection-string "$CONNECTION_STRING" \
                --overwrite \
                --no-progress
        fi
    done
    
    print_success "Performance test files uploaded"
    print_status "Monitor the function logs to see performance with multiple files"
}

# Show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  full-test        Run complete test scenario (default)"
    echo "  create-data      Create comprehensive test data only"
    echo "  upload           Upload test files to containers"
    echo "  monitor          Monitor function execution and results"
    echo "  validate         Validate test results"
    echo "  performance      Run performance test with multiple files"
    echo "  cleanup          Clean up test data and containers"
    echo "  help             Show this help message"
    echo ""
    echo "Full test workflow:"
    echo "  1. Creates comprehensive test data (PNG and non-PNG files)"
    echo "  2. Sets up blob containers"
    echo "  3. Uploads all test files"
    echo "  4. Monitors function execution"
    echo "  5. Validates results"
    echo ""
    echo "Prerequisites:"
    echo "  - Azurite running on localhost:10000"
    echo "  - Azure CLI installed"
    echo "  - Azure Function running (npm start)"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Azurite is running
    if ! nc -z localhost 10000 2>/dev/null; then
        print_error "Azurite is not running. Start it with:"
        echo "azurite --silent --location ./azurite-data --debug ./azurite-debug.log"
        exit 1
    fi
    
    # Check if Azure CLI is available
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI is not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Main command handling
main() {
    case "${1:-full-test}" in
        "full-test")
            check_prerequisites
            create_comprehensive_test_data
            setup_containers
            upload_all_test_files
            monitor_function
            validate_results
            ;;
        "create-data")
            create_comprehensive_test_data
            ;;
        "upload")
            check_prerequisites
            setup_containers
            upload_all_test_files
            ;;
        "monitor")
            check_prerequisites
            monitor_function
            ;;
        "validate")
            check_prerequisites
            validate_results
            ;;
        "performance")
            check_prerequisites
            run_performance_test
            ;;
        "cleanup")
            cleanup_test_data
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

main "$@"
