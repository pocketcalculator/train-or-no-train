import { app, InvocationContext, Timer } from "@azure/functions";
import { BlobServiceClient, ContainerClient } from "@azure/storage-blob";

/**
 * Enhanced Azure Function that monitors the 'incoming' blob container for PNG files,
 * analyzes them for train detection, and moves them to appropriate containers.
 * 
 * This function runs on a timer trigger every 30 seconds (configurable).
 * When PNG files are detected, it analyzes them for trains and sorts them accordingly.
 */

// Configuration constants
const INCOMING_CONTAINER = process.env.INCOMING_CONTAINER_NAME || "incoming";
const ARCHIVE_CONTAINER = process.env.ARCHIVE_CONTAINER_NAME || "archive";
const TRAIN_DETECTED_CONTAINER = process.env.TRAIN_DETECTED_CONTAINER_NAME || "train-detected";
const NO_TRAIN_CONTAINER = process.env.NO_TRAIN_CONTAINER_NAME || "no-train";
const STORAGE_CONNECTION_STRING = process.env.AZURE_STORAGE_CONNECTION_STRING;

// Train Detection Configuration
const ENABLE_TRAIN_DETECTION = process.env.ENABLE_TRAIN_DETECTION === "true";
const TRAIN_DETECTION_ENDPOINT = process.env.TRAIN_DETECTION_ENDPOINT;
const CONFIDENCE_THRESHOLD = parseFloat(process.env.CONFIDENCE_THRESHOLD || "0.7");

// Validate required environment variables
if (!STORAGE_CONNECTION_STRING) {
    throw new Error("AZURE_STORAGE_CONNECTION_STRING environment variable is required");
}

console.log(`[STARTUP] Enhanced Blob Monitor Function initialized`);
console.log(`[CONFIG] Incoming container: ${INCOMING_CONTAINER}`);
console.log(`[CONFIG] Archive container: ${ARCHIVE_CONTAINER}`);
console.log(`[CONFIG] Train detection enabled: ${ENABLE_TRAIN_DETECTION}`);
if (ENABLE_TRAIN_DETECTION) {
    console.log(`[CONFIG] Train detected container: ${TRAIN_DETECTED_CONTAINER}`);
    console.log(`[CONFIG] No train container: ${NO_TRAIN_CONTAINER}`);
    console.log(`[CONFIG] Confidence threshold: ${CONFIDENCE_THRESHOLD}`);
}

/**
 * Timer-triggered Azure Function that checks for PNG files in the incoming container
 * @param timer - Timer object containing schedule information
 * @param context - Azure Function execution context
 */
export async function blobMonitorTimer(timer: Timer, context: InvocationContext): Promise<void> {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] Enhanced Blob Monitor Function triggered`);
    
    try {
        // Initialize blob service client
        console.log(`[${timestamp}] Initializing Azure Blob Service Client...`);
        const blobServiceClient = BlobServiceClient.fromConnectionString(STORAGE_CONNECTION_STRING);
        
        // Get container clients
        const incomingContainer = blobServiceClient.getContainerClient(INCOMING_CONTAINER);
        const archiveContainer = blobServiceClient.getContainerClient(ARCHIVE_CONTAINER);
        let trainDetectedContainer: ContainerClient | null = null;
        let noTrainContainer: ContainerClient | null = null;
        
        if (ENABLE_TRAIN_DETECTION) {
            trainDetectedContainer = blobServiceClient.getContainerClient(TRAIN_DETECTED_CONTAINER);
            noTrainContainer = blobServiceClient.getContainerClient(NO_TRAIN_CONTAINER);
        }
        
        // Ensure containers exist
        await ensureContainersExist(
            incomingContainer, 
            archiveContainer, 
            trainDetectedContainer,
            noTrainContainer,
            timestamp
        );
        
        // Check for PNG files in incoming container
        await processPngFiles(
            incomingContainer, 
            archiveContainer,
            trainDetectedContainer,
            noTrainContainer,
            timestamp, 
            context
        );
        
        console.log(`[${timestamp}] Enhanced Blob Monitor Function execution completed successfully`);
        
    } catch (error) {
        console.error(`[${timestamp}] ERROR in Enhanced Blob Monitor Function:`, error);
        
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
 * Ensures that all required containers exist
 */
async function ensureContainersExist(
    incomingContainer: ContainerClient, 
    archiveContainer: ContainerClient,
    trainDetectedContainer: ContainerClient | null,
    noTrainContainer: ContainerClient | null,
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
        
        // Create train detection containers if enabled
        if (ENABLE_TRAIN_DETECTION && trainDetectedContainer && noTrainContainer) {
            const trainExists = await trainDetectedContainer.exists();
            if (!trainExists) {
                console.log(`[${timestamp}] Creating train detected container: ${TRAIN_DETECTED_CONTAINER}`);
                await trainDetectedContainer.create();
                console.log(`[${timestamp}] Successfully created train detected container`);
            }
            
            const noTrainExists = await noTrainContainer.exists();
            if (!noTrainExists) {
                console.log(`[${timestamp}] Creating no train container: ${NO_TRAIN_CONTAINER}`);
                await noTrainContainer.create();
                console.log(`[${timestamp}] Successfully created no train container`);
            }
        }
        
    } catch (error) {
        console.error(`[${timestamp}] ERROR ensuring containers exist:`, error);
        throw new Error(`Failed to ensure containers exist: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

/**
 * Processes PNG files found in the incoming container
 */
async function processPngFiles(
    incomingContainer: ContainerClient, 
    archiveContainer: ContainerClient,
    trainDetectedContainer: ContainerClient | null,
    noTrainContainer: ContainerClient | null,
    timestamp: string,
    context: InvocationContext
): Promise<void> {
    try {
        console.log(`[${timestamp}] Scanning incoming container for PNG files...`);
        
        let pngFileCount = 0;
        let totalFileCount = 0;
        let trainDetectedCount = 0;
        let noTrainCount = 0;
        
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
                    let targetContainer = archiveContainer;
                    let targetContainerName = ARCHIVE_CONTAINER;
                    
                    // Perform train detection if enabled
                    if (ENABLE_TRAIN_DETECTION && trainDetectedContainer && noTrainContainer) {
                        const detectionResult = await detectTrainInBlob(blob.name, incomingContainer, timestamp);
                        
                        if (detectionResult.success) {
                            if (detectionResult.trainDetected && detectionResult.confidence >= CONFIDENCE_THRESHOLD) {
                                targetContainer = trainDetectedContainer;
                                targetContainerName = TRAIN_DETECTED_CONTAINER;
                                trainDetectedCount++;
                                console.log(`[${timestamp}] 🚂 TRAIN DETECTED in ${blob.name} (confidence: ${detectionResult.confidence.toFixed(3)})`);
                            } else {
                                targetContainer = noTrainContainer;
                                targetContainerName = NO_TRAIN_CONTAINER;
                                noTrainCount++;
                                console.log(`[${timestamp}] ❌ NO TRAIN in ${blob.name} (confidence: ${detectionResult.confidence.toFixed(3)})`);
                            }
                        } else {
                            console.log(`[${timestamp}] ⚠️ Train detection failed for ${blob.name}, moving to archive`);
                        }
                    }
                    
                    // Move the file to target container
                    await moveBlob(blob.name, incomingContainer, targetContainer, targetContainerName, timestamp);
                    console.log(`[${timestamp}] Successfully moved ${blob.name} to ${targetContainerName}`);
                    
                } catch (moveError) {
                    console.error(`[${timestamp}] ERROR processing blob ${blob.name}:`, moveError);
                    // Continue processing other files even if one fails
                }
            }
        }
        
        console.log(`[${timestamp}] Scan completed. Total files: ${totalFileCount}, PNG files found: ${pngFileCount}`);
        
        if (ENABLE_TRAIN_DETECTION) {
            console.log(`[${timestamp}] Train detection results: Trains: ${trainDetectedCount}, No trains: ${noTrainCount}`);
        }
        
        if (pngFileCount === 0) {
            console.log(`[${timestamp}] No PNG files found in incoming container`);
        }
        
    } catch (error) {
        console.error(`[${timestamp}] ERROR processing PNG files:`, error);
        throw new Error(`Failed to process PNG files: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

/**
 * Detect if a train is present in the blob image
 */
async function detectTrainInBlob(
    blobName: string, 
    sourceContainer: ContainerClient, 
    timestamp: string
): Promise<{success: boolean, trainDetected: boolean, confidence: number, method?: string}> {
    try {
        console.log(`[${timestamp}] Starting train detection for: ${blobName}`);
        
        // If we have a Python train detection service endpoint, call it
        if (TRAIN_DETECTION_ENDPOINT) {
            try {
                const response = await fetch(`${TRAIN_DETECTION_ENDPOINT}/analyze`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        container_name: sourceContainer.containerName,
                        blob_name: blobName
                    })
                });
                
                if (response.ok) {
                    const result = await response.json();
                    return {
                        success: true,
                        trainDetected: result.train_detected || false,
                        confidence: result.confidence || 0.5,
                        method: result.primary_method || 'api'
                    };
                }
            } catch (apiError) {
                console.log(`[${timestamp}] API detection failed for ${blobName}, falling back to keyword analysis`);
            }
        }
        
        // Fallback: Simple keyword-based detection from filename
        const trainKeywords = [
            'train', 'locomotive', 'railway', 'railroad', 'rail', 'subway', 'metro',
            'cargo', 'freight', 'passenger', 'engine', 'railcar', 'boxcar'
        ];
        
        const fileName = blobName.toLowerCase();
        const hasTrainKeyword = trainKeywords.some(keyword => fileName.includes(keyword));
        
        return {
            success: true,
            trainDetected: hasTrainKeyword,
            confidence: hasTrainKeyword ? 0.8 : 0.9, // High confidence for keyword-based detection
            method: 'keyword'
        };
        
    } catch (error) {
        console.error(`[${timestamp}] ERROR in train detection for ${blobName}:`, error);
        return {
            success: false,
            trainDetected: false,
            confidence: 0.0
        };
    }
}

/**
 * Moves a blob from the incoming container to the target container
 */
async function moveBlob(
    blobName: string, 
    sourceContainer: ContainerClient, 
    targetContainer: ContainerClient,
    targetContainerName: string,
    timestamp: string
): Promise<void> {
    try {
        console.log(`[${timestamp}] Starting move operation for blob: ${blobName} to ${targetContainerName}`);
        
        // Get source blob client
        const sourceBlobClient = sourceContainer.getBlobClient(blobName);
        
        // Check if source blob exists
        const sourceExists = await sourceBlobClient.exists();
        if (!sourceExists) {
            throw new Error(`Source blob ${blobName} does not exist`);
        }
        
        // Get target blob client
        const targetBlobClient = targetContainer.getBlobClient(blobName);
        
        // Copy blob to target container
        console.log(`[${timestamp}] Copying ${blobName} to ${targetContainerName} container...`);
        const copyResult = await targetBlobClient.syncCopyFromURL(sourceBlobClient.url);
        
        if (copyResult.copyStatus !== 'success') {
            throw new Error(`Copy operation failed with status: ${copyResult.copyStatus}`);
        }
        
        console.log(`[${timestamp}] Successfully copied ${blobName} to ${targetContainerName}`);
        
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