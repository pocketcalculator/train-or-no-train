using 'main.bicep'

// Parameters for the blob monitor function infrastructure
param baseName = 'blobmonitor'
param environment = 'dev'
param location = 'East US'
param functionAppSkuName = 'Y1'
param storageSkuName = 'Standard_LRS'
