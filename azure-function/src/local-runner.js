#!/usr/bin/env node

/**
 * Environment loader for Azure Functions local development
 * Loads settings from local.settings.json and starts the test runner
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// Load local.settings.json
const settingsPath = path.join(__dirname, '..', 'local.settings.json');

try {
    if (fs.existsSync(settingsPath)) {
        const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
        
        // Set environment variables from local.settings.json
        if (settings.Values) {
            for (const [key, value] of Object.entries(settings.Values)) {
                process.env[key] = value;
            }
            console.log('✅ Loaded environment variables from local.settings.json');
        }
    } else {
        console.warn('⚠️  local.settings.json not found, using default environment variables');
    }
} catch (error) {
    console.error('❌ Error loading local.settings.json:', error.message);
    process.exit(1);
}

// Start the test runner
console.log('🚀 Starting Azure Function test runner...\n');

const testRunner = spawn('node', [path.join(__dirname, '..', 'dist', 'src', 'test-runner.js')], {
    stdio: 'inherit',
    env: process.env
});

testRunner.on('close', (code) => {
    console.log(`\n🏁 Test runner exited with code ${code}`);
    process.exit(code);
});

testRunner.on('error', (err) => {
    console.error('❌ Failed to start test runner:', err);
    process.exit(1);
});

// Handle signals
process.on('SIGINT', () => {
    console.log('\n🛑 Stopping test runner...');
    testRunner.kill('SIGINT');
});

process.on('SIGTERM', () => {
    console.log('\n🛑 Stopping test runner...');
    testRunner.kill('SIGTERM');
});
