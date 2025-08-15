import { app, InvocationContext, Timer } from "@azure/functions";
import { BlobServiceClient, ContainerClient } from "@azure/storage-blob";

/**
 * Azure Function that monitors the 'incoming' blob container for PNG files
 * and moves them to the 'archive' container when found.
 * 
 * This function runs on a timer trigger every 30 seconds (configurable).
 * When PNG files are detected, it logs their discovery and moves them to archive.
 */

// Configuration constants
const INCOMING_CONTAINER = process.env.INCOMING_CONTAINER_NAME || "incoming";
const ARCHIVE_CONTAINER = process.env.ARCHIVE_CONTAINER_NAME || "archive";
const STORAGE_CONNECTION_STRING = process.env.AZURE_STORAGE_CONNECTION_STRING;

// Validate required environment variables
if (!STORAGE_CONNECTION_STRING) {
    throw new Error("AZURE_STORAGE_CONNECTION_STRING environment variable is required");
}

console.log(`[STARTUP] Blob Monitor Function initialized`);
console.log(`[CONFIG] Incoming container: ${INCOMING_CONTAINER}`);
console.log(`[CONFIG] Archive container: ${ARCHIVE_CONTAINER}`);

/**
 * Timer-triggered Azure Function that checks for PNG files in the incoming container
 * @param timer - Timer object containing schedule information
 * @param context - Azure Function execution context
 */
export async function blobMonitorTimer(timer: Timer, context: InvocationContext): Promise<void> {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] Blob Monitor Function triggered`);
    
    try {
        // Initialize blob service client
        console.log(`[${timestamp}] Initializing Azure Blob Service Client...`);
        const blobServiceClient = BlobServiceClient.fromConnectionString(STORAGE_CONNECTION_STRING);
        
        // Get container clients
        const incomingContainer = blobServiceClient.getContainerClient(INCOMING_CONTAINER);
        const archiveContainer = blobServiceClient.getContainerClient(ARCHIVE_CONTAINER);
        
        // Ensure containers exist
        await ensureContainersExist(incomingContainer, archiveContainer, timestamp);
        
        // Check for PNG files in incoming container
        await processPngFiles(incomingContainer, archiveContainer, timestamp, context);
        
        console.log(`[${timestamp}] Blob Monitor Function execution completed successfully`);
        
    } catch (error) {
        console.error(`[${timestamp}] ERROR in Blob Monitor Function:`, error);
        
        // Log additional error details for troubleshooting
        if (error instanceof Error) {
            console.error(`[${timestamp}] Error name: ${error.name}`);
            console.error(`[${timestamp}] Error message: ${error.message}`);
            console.error(`[${timestamp}] Error stack: ${error.stack}`);
        }
        
        // Re-throw to ensure function is marked as failed
        throw error;
    }
}

/**
 * Ensures that both incoming and archive containers exist
 * @param incomingContainer - Incoming container client
 * @param archiveContainer - Archive container client
 * @param timestamp - Current timestamp for logging
 */
async function ensureContainersExist(
    incomingContainer: ContainerClient, 
    archiveContainer: ContainerClient, 
    timestamp: string
): Promise<void> {
    try {
        console.log(`[${timestamp}] Checking if containers exist...`);
        
        // Check and create incoming container if needed
        const incomingExists = await incomingContainer.exists();
        if (!incomingExists) {
            console.log(`[${timestamp}] Creating incoming container: ${INCOMING_CONTAINER}`);
            await incomingContainer.create();
            console.log(`[${timestamp}] Successfully created incoming container`);
        } else {
            console.log(`[${timestamp}] Incoming container exists: ${INCOMING_CONTAINER}`);
        }
        
        // Check and create archive container if needed
        const archiveExists = await archiveContainer.exists();
        if (!archiveExists) {
            console.log(`[${timestamp}] Creating archive container: ${ARCHIVE_CONTAINER}`);
            await archiveContainer.create();
            console.log(`[${timestamp}] Successfully created archive container`);
        } else {
            console.log(`[${timestamp}] Archive container exists: ${ARCHIVE_CONTAINER}`);
        }
        
    } catch (error) {
        console.error(`[${timestamp}] ERROR ensuring containers exist:`, error);
        throw new Error(`Failed to ensure containers exist: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

/**
 * Processes PNG files found in the incoming container
 * @param incomingContainer - Incoming container client
 * @param archiveContainer - Archive container client
 * @param timestamp - Current timestamp for logging
 * @param context - Azure Function execution context
 */
async function processPngFiles(
    incomingContainer: ContainerClient, 
    archiveContainer: ContainerClient, 
    timestamp: string,
    context: InvocationContext
): Promise<void> {
    try {
        console.log(`[${timestamp}] Scanning incoming container for PNG files...`);
        
        let pngFileCount = 0;
        let totalFileCount = 0;
        
        // List all blobs in the incoming container
        for await (const blob of incomingContainer.listBlobsFlat()) {
            totalFileCount++;
            
            // Check if the blob is a PNG file (case-insensitive)
            if (blob.name.toLowerCase().endsWith('.png')) {
                pngFileCount++;
                console.log(`[${timestamp}] PNG file detected: ${blob.name}`);
                
                // Log the required message to stdout
                console.log(`${blob.name} found at ${timestamp}`);
                
                try {
                    // Move the file to archive container
                    await moveBlob(blob.name, incomingContainer, archiveContainer, timestamp);
                    console.log(`[${timestamp}] Successfully moved ${blob.name} to archive`);
                    
                } catch (moveError) {
                    console.error(`[${timestamp}] ERROR moving blob ${blob.name}:`, moveError);
                    // Continue processing other files even if one fails
                }
            }
        }
        
        console.log(`[${timestamp}] Scan completed. Total files: ${totalFileCount}, PNG files found: ${pngFileCount}`);
        
        if (pngFileCount === 0) {
            console.log(`[${timestamp}] No PNG files found in incoming container`);
        }
        
    } catch (error) {
        console.error(`[${timestamp}] ERROR processing PNG files:`, error);
        throw new Error(`Failed to process PNG files: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

/**
 * Moves a blob from the incoming container to the archive container
 * @param blobName - Name of the blob to move
 * @param sourceContainer - Source container client
 * @param targetContainer - Target container client
 * @param timestamp - Current timestamp for logging
 */
async function moveBlob(
    blobName: string, 
    sourceContainer: ContainerClient, 
    targetContainer: ContainerClient, 
    timestamp: string
): Promise<void> {
    try {
        console.log(`[${timestamp}] Starting move operation for blob: ${blobName}`);
        
        // Get source blob client
        const sourceBlobClient = sourceContainer.getBlobClient(blobName);
        
        // Check if source blob exists
        const sourceExists = await sourceBlobClient.exists();
        if (!sourceExists) {
            throw new Error(`Source blob ${blobName} does not exist`);
        }
        
        // Get target blob client
        const targetBlobClient = targetContainer.getBlobClient(blobName);
        
        // Copy blob to archive container
        console.log(`[${timestamp}] Copying ${blobName} to archive container...`);
        const copyResult = await targetBlobClient.syncCopyFromURL(sourceBlobClient.url);
        
        if (copyResult.copyStatus !== 'success') {
            throw new Error(`Copy operation failed with status: ${copyResult.copyStatus}`);
        }
        
        console.log(`[${timestamp}] Successfully copied ${blobName} to archive`);
        
        // Delete original blob from incoming container
        console.log(`[${timestamp}] Deleting ${blobName} from incoming container...`);
        await sourceBlobClient.delete();
        
        console.log(`[${timestamp}] Successfully deleted ${blobName} from incoming container`);
        console.log(`[${timestamp}] Move operation completed for ${blobName}`);
        
    } catch (error) {
        console.error(`[${timestamp}] ERROR in move operation for ${blobName}:`, error);
        throw new Error(`Failed to move blob ${blobName}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

// Register the function with Azure Functions runtime
// Timer trigger runs every 30 seconds (0 */30 * * * * = every 30 seconds)
// You can modify the schedule as needed:
// - "0 */1 * * * *" = every minute
// - "0 */5 * * * *" = every 5 minutes
// - "0 0 */1 * * *" = every hour
app.timer('blobMonitorTimer', {
    schedule: '0 */30 * * * *', // Every 30 seconds
    handler: blobMonitorTimer
});
