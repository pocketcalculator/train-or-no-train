#!/usr/bin/env node

/**
 * Local test runner for the Azure Blob Monitor Function
 * This simulates the Azure Functions timer trigger for local testing
 */

import { blobMonitorTimer } from './functions/blobMonitor.js';
import type { Timer, InvocationContext } from '@azure/functions';

// Mock Azure Functions context and timer objects
const mockTimer = {
    isPastDue: false,
    schedule: {
        adjustForDST: false
    },
    scheduleStatus: {
        last: new Date().toISOString(),
        lastUpdated: new Date().toISOString(),
        next: new Date(Date.now() + 30000).toISOString() // 30 seconds from now
    }
};

const mockContext = {
    invocationId: 'test-invocation-' + Math.random().toString(36).substr(2, 9),
    functionName: 'blobMonitorTimer',
    executionContext: {
        invocationId: 'test-invocation-' + Math.random().toString(36).substr(2, 9),
        functionName: 'blobMonitorTimer',
        functionDirectory: process.cwd()
    },
    extraInputs: new Map(),
    extraOutputs: new Map(),
    options: {},
    log: console.log,
    error: console.error,
    warn: console.warn,
    info: console.info,
    debug: console.debug,
    trace: console.trace
};

console.log('=== Azure Blob Monitor Function - Local Test Runner ===');
console.log('This simulates the Azure Functions timer trigger for testing');
console.log('Press Ctrl+C to stop\n');

// Function to run the blob monitor
async function runBlobMonitor() {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ⏰ Timer trigger activated (simulated)`);
    
    try {
        await blobMonitorTimer(mockTimer as unknown as Timer, mockContext as unknown as InvocationContext);
        console.log(`[${timestamp}] ✅ Function execution completed successfully\n`);
    } catch (error) {
        console.error(`[${timestamp}] ❌ Function execution failed:`, error);
        console.error(`[${timestamp}] Error details:`, error.message, '\n');
    }
}

// Run immediately, then every 30 seconds
console.log('Starting blob monitor function...');
runBlobMonitor();

const interval = setInterval(runBlobMonitor, 30000); // 30 seconds

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Stopping blob monitor function...');
    clearInterval(interval);
    console.log('✅ Stopped successfully');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n🛑 Stopping blob monitor function...');
    clearInterval(interval);
    console.log('✅ Stopped successfully');
    process.exit(0);
});
