#!/bin/bash

# Azure deployment script for the Blob Monitor Function
# This script deploys the infrastructure and function code to Azure

set -e

# Configuration
RESOURCE_GROUP_NAME="rg-blobmonitor-dev"
LOCATION="East US"
SUBSCRIPTION_ID=""  # Set your subscription ID here
BICEP_FILE="./infrastructure/main.bicep"
BICEP_PARAMS_FILE="./infrastructure/main.bicepparam"

# Colors for output
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

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI is not installed. Please install it first."
        echo "Install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi
    
    # Check if logged in to Azure
    if ! az account show &> /dev/null; then
        print_error "Not logged in to Azure. Please run: az login"
        exit 1
    fi
    
    # Check if Functions Core Tools is available
    if ! command -v func &> /dev/null; then
        print_warning "Azure Functions Core Tools not found. Installing..."
        npm install -g azure-functions-core-tools@4 --unsafe-perm true
    fi
    
    print_success "Prerequisites check completed"
}

# Set subscription
set_subscription() {
    if [ -n "$SUBSCRIPTION_ID" ]; then
        print_status "Setting subscription to: $SUBSCRIPTION_ID"
        az account set --subscription "$SUBSCRIPTION_ID"
        print_success "Subscription set"
    else
        print_warning "No subscription ID provided. Using current subscription:"
        az account show --query "name" -o tsv
    fi
}

# Create resource group
create_resource_group() {
    print_status "Creating resource group: $RESOURCE_GROUP_NAME"
    
    if az group show --name "$RESOURCE_GROUP_NAME" &> /dev/null; then
        print_warning "Resource group already exists"
    else
        az group create --name "$RESOURCE_GROUP_NAME" --location "$LOCATION"
        print_success "Resource group created"
    fi
}

# Deploy infrastructure
deploy_infrastructure() {
    print_status "Deploying infrastructure using Bicep..."
    
    # Validate the Bicep template first
    print_status "Validating Bicep template..."
    az deployment group validate \
        --resource-group "$RESOURCE_GROUP_NAME" \
        --template-file "$BICEP_FILE" \
        --parameters "$BICEP_PARAMS_FILE"
    
    print_success "Bicep template validation passed"
    
    # Deploy the infrastructure
    print_status "Deploying infrastructure..."
    DEPLOYMENT_OUTPUT=$(az deployment group create \
        --resource-group "$RESOURCE_GROUP_NAME" \
        --template-file "$BICEP_FILE" \
        --parameters "$BICEP_PARAMS_FILE" \
        --query 'properties.outputs' \
        -o json)
    
    if [ $? -eq 0 ]; then
        print_success "Infrastructure deployment completed"
        
        # Extract outputs
        FUNCTION_APP_NAME=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.functionAppName.value')
        STORAGE_ACCOUNT_NAME=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.storageAccountName.value')
        
        echo ""
        print_success "Deployment outputs:"
        echo "Function App Name: $FUNCTION_APP_NAME"
        echo "Storage Account Name: $STORAGE_ACCOUNT_NAME"
        echo ""
        
        # Store outputs for function deployment
        echo "FUNCTION_APP_NAME=$FUNCTION_APP_NAME" > .deployment-outputs
        echo "STORAGE_ACCOUNT_NAME=$STORAGE_ACCOUNT_NAME" >> .deployment-outputs
        
    else
        print_error "Infrastructure deployment failed"
        exit 1
    fi
}

# Build and deploy function
deploy_function() {
    print_status "Building and deploying function code..."
    
    # Load deployment outputs
    if [ -f ".deployment-outputs" ]; then
        source .deployment-outputs
    else
        print_error "Deployment outputs not found. Run infrastructure deployment first."
        exit 1
    fi
    
    # Build the project
    print_status "Building TypeScript project..."
    npm run build
    
    # Deploy the function
    print_status "Deploying function to: $FUNCTION_APP_NAME"
    func azure functionapp publish "$FUNCTION_APP_NAME" --typescript
    
    if [ $? -eq 0 ]; then
        print_success "Function deployment completed"
        echo ""
        print_success "Function app URL: https://$FUNCTION_APP_NAME.azurewebsites.net"
    else
        print_error "Function deployment failed"
        exit 1
    fi
}

# Test deployment
test_deployment() {
    print_status "Testing deployment..."
    
    if [ -f ".deployment-outputs" ]; then
        source .deployment-outputs
    else
        print_error "Deployment outputs not found."
        exit 1
    fi
    
    # Test function app responsiveness
    print_status "Testing function app connectivity..."
    
    FUNCTION_URL="https://$FUNCTION_APP_NAME.azurewebsites.net"
    if curl -s --head "$FUNCTION_URL" | head -n 1 | grep -q "200 OK"; then
        print_success "Function app is responsive"
    else
        print_warning "Function app may still be starting up"
    fi
    
    print_status "Deployment testing completed"
    echo ""
    print_success "Next steps:"
    echo "1. Upload PNG files to the 'incoming' container in storage account: $STORAGE_ACCOUNT_NAME"
    echo "2. Monitor function logs: az functionapp logs tail --name $FUNCTION_APP_NAME --resource-group $RESOURCE_GROUP_NAME"
    echo "3. Check Application Insights for telemetry and performance data"
}

# Cleanup function
cleanup() {
    print_warning "Cleaning up deployment artifacts..."
    rm -f .deployment-outputs
}

# Show help
show_help() {
    echo "Azure Blob Monitor Function Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  all              Deploy infrastructure and function (default)"
    echo "  infra            Deploy infrastructure only"
    echo "  function         Deploy function code only"
    echo "  test             Test deployment"
    echo "  cleanup          Clean up deployment artifacts"
    echo "  help             Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  SUBSCRIPTION_ID  Azure subscription ID (optional)"
    echo ""
    echo "Before running, ensure you have:"
    echo "  - Azure CLI installed and logged in"
    echo "  - Node.js and npm installed"
    echo "  - Built the TypeScript project (npm run build)"
}

# Main execution
main() {
    case "${1:-all}" in
        "all")
            check_prerequisites
            set_subscription
            create_resource_group
            deploy_infrastructure
            deploy_function
            test_deployment
            ;;
        "infra")
            check_prerequisites
            set_subscription
            create_resource_group
            deploy_infrastructure
            ;;
        "function")
            check_prerequisites
            deploy_function
            ;;
        "test")
            test_deployment
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Trap cleanup on script exit
trap cleanup EXIT

# Run main function
main "$@"
