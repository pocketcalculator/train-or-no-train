#!/bin/bash

# Azure Best Practices Validation Script
# Validates that the implementation follows Azure development best practices

echo "=== Azure Best Practices Validation ==="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

passed=0
warnings=0
failed=0

check_pass() {
    echo -e "${GREEN}✅ PASS:${NC} $1"
    ((passed++))
}

check_warning() {
    echo -e "${YELLOW}⚠️  WARNING:${NC} $1"
    ((warnings++))
}

check_fail() {
    echo -e "${RED}❌ FAIL:${NC} $1"
    ((failed++))
}

echo "Validating Azure Function implementation against best practices..."
echo ""

# 1. Check Node.js version
echo "🔍 Checking Node.js Version..."
if grep -q '"node": ">=20.0.0"' package.json; then
    check_pass "Using Node.js 20+ (latest supported by Azure Functions)"
else
    check_warning "Consider using Node.js 20+ for better performance and security"
fi

# 2. Check Azure Functions version
echo ""
echo "🔍 Checking Azure Functions Version..."
if grep -q '"@azure/functions": "\^4' package.json; then
    check_pass "Using Azure Functions v4 (latest version)"
else
    check_warning "Consider upgrading to Azure Functions v4"
fi

# 3. Check error handling
echo ""
echo "🔍 Checking Error Handling..."
if grep -q "try {" src/functions/blobMonitor.ts && grep -q "catch" src/functions/blobMonitor.ts; then
    check_pass "Proper try-catch error handling implemented"
else
    check_fail "Missing proper error handling in function code"
fi

# 4. Check logging
echo ""
echo "🔍 Checking Logging Implementation..."
if grep -q "console.log" src/functions/blobMonitor.ts && grep -q "console.error" src/functions/blobMonitor.ts; then
    check_pass "Comprehensive logging implemented for troubleshooting"
else
    check_warning "Consider adding more comprehensive logging"
fi

# 5. Check environment variable usage
echo ""
echo "🔍 Checking Environment Variables..."
if grep -q "process.env" src/functions/blobMonitor.ts; then
    check_pass "Using environment variables for configuration"
else
    check_fail "Missing environment variable configuration"
fi

# 6. Check connection string security
echo ""
echo "🔍 Checking Security Practices..."
if grep -q "AZURE_STORAGE_CONNECTION_STRING" local.settings.json; then
    check_pass "Using connection strings through environment variables"
else
    check_warning "Ensure connection strings are not hardcoded"
fi

# 7. Check TypeScript usage
echo ""
echo "🔍 Checking TypeScript Implementation..."
if [ -f "tsconfig.json" ] && [ -f "src/functions/blobMonitor.ts" ]; then
    check_pass "Using TypeScript for better type safety and development experience"
else
    check_warning "Consider using TypeScript for better development experience"
fi

# 8. Check Bicep infrastructure
echo ""
echo "🔍 Checking Infrastructure as Code..."
if [ -f "infrastructure/main.bicep" ]; then
    check_pass "Using Bicep for Infrastructure as Code"
else
    check_warning "Consider using Bicep or ARM templates for infrastructure"
fi

# 9. Check security in Bicep
echo ""
echo "🔍 Checking Bicep Security..."
if grep -q "httpsOnly: true" infrastructure/main.bicep; then
    check_pass "HTTPS-only enforcement in Bicep template"
else
    check_fail "Missing HTTPS-only enforcement"
fi

if grep -q "minimumTlsVersion" infrastructure/main.bicep; then
    check_pass "TLS version enforcement in storage account"
else
    check_fail "Missing TLS version enforcement"
fi

# 10. Check monitoring setup
echo ""
echo "🔍 Checking Monitoring and Observability..."
if grep -q "applicationInsights" infrastructure/main.bicep; then
    check_pass "Application Insights configured for monitoring"
else
    check_warning "Consider adding Application Insights for monitoring"
fi

# 11. Check container security
echo ""
echo "🔍 Checking Storage Container Security..."
if grep -q "publicAccess: 'None'" infrastructure/main.bicep; then
    check_pass "Blob containers configured with no public access"
else
    check_fail "Missing proper container security configuration"
fi

# 12. Check function timeout
echo ""
echo "🔍 Checking Function Configuration..."
if grep -q "functionTimeout" host.json; then
    check_pass "Function timeout properly configured"
else
    check_warning "Consider configuring function timeout in host.json"
fi

# 13. Check deployment scripts
echo ""
echo "🔍 Checking Deployment Automation..."
if [ -f "deploy-to-azure.sh" ] && [ -x "deploy-to-azure.sh" ]; then
    check_pass "Automated deployment script provided"
else
    check_warning "Consider adding automated deployment scripts"
fi

# 14. Check testing infrastructure
echo ""
echo "🔍 Checking Testing Infrastructure..."
if [ -d "test-scripts" ] && [ -f "test-scripts/comprehensive-test.sh" ]; then
    check_pass "Comprehensive testing scripts provided"
else
    check_warning "Consider adding automated testing scripts"
fi

# 15. Check documentation
echo ""
echo "🔍 Checking Documentation..."
if [ -f "README.md" ] && [ -s "README.md" ]; then
    check_pass "Comprehensive README documentation provided"
else
    check_warning "Consider adding comprehensive documentation"
fi

# 16. Check .gitignore
echo ""
echo "🔍 Checking Source Control..."
if [ -f ".gitignore" ] && grep -q "local.settings.json" .gitignore; then
    check_pass "Proper .gitignore configuration for Azure Functions"
else
    check_warning "Consider improving .gitignore for security"
fi

# 17. Check function naming and structure
echo ""
echo "🔍 Checking Function Structure..."
if grep -q "app.timer" src/functions/blobMonitor.ts; then
    check_pass "Using proper Azure Functions v4 programming model"
else
    check_warning "Consider using Azure Functions v4 programming model"
fi

# 18. Check blob operations
echo ""
echo "🔍 Checking Blob Operations..."
if grep -q "syncCopyFromURL" src/functions/blobMonitor.ts && grep -q "delete" src/functions/blobMonitor.ts; then
    check_pass "Proper blob copy and delete operations implemented"
else
    check_warning "Ensure proper blob move operations (copy then delete)"
fi

# 19. Check parameter files
echo ""
echo "🔍 Checking Bicep Parameters..."
if [ -f "infrastructure/main.bicepparam" ]; then
    check_pass "Using Bicep parameter files for configuration"
else
    check_warning "Consider using Bicep parameter files"
fi

# 20. Check resource naming
echo ""
echo "🔍 Checking Resource Naming..."
if grep -q "resourceNaming" infrastructure/main.bicep; then
    check_pass "Consistent resource naming convention implemented"
else
    check_warning "Consider implementing consistent resource naming"
fi

# Summary
echo ""
echo "========================================="
echo "AZURE BEST PRACTICES VALIDATION SUMMARY"
echo "========================================="
echo -e "${GREEN}Passed: $passed${NC}"
echo -e "${YELLOW}Warnings: $warnings${NC}"
echo -e "${RED}Failed: $failed${NC}"
echo ""

if [ $failed -eq 0 ]; then
    if [ $warnings -eq 0 ]; then
        echo -e "${GREEN}🎉 EXCELLENT: All Azure best practices are followed!${NC}"
        exit 0
    else
        echo -e "${YELLOW}✅ GOOD: Core best practices followed with some suggestions for improvement.${NC}"
        exit 0
    fi
else
    echo -e "${RED}⚠️  NEEDS IMPROVEMENT: Some critical best practices are missing.${NC}"
    echo "Please address the failed checks before deploying to production."
    exit 1
fi
