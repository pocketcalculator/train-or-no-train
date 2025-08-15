// Azure infrastructure template for the Blob Monitor Function
// This template creates all required resources following Azure best practices

@description('The base name for all resources (will be used as prefix)')
param baseName string = 'blobmonitor'

@description('The Azure region for all resources')
param location string = resourceGroup().location

@description('The environment (dev, test, prod)')
@allowed(['dev', 'test', 'prod'])
param environment string = 'dev'

@description('The SKU for the Function App hosting plan')
@allowed(['Y1', 'EP1', 'EP2', 'EP3'])
param functionAppSkuName string = 'Y1'

@description('The storage account SKU')
@allowed(['Standard_LRS', 'Standard_GRS', 'Standard_RAGRS'])
param storageSkuName string = 'Standard_LRS'

// Variables for naming consistency
var resourceNaming = {
  storageAccount: '${baseName}${environment}st'
  functionApp: '${baseName}-${environment}-func'
  hostingPlan: '${baseName}-${environment}-plan'
  applicationInsights: '${baseName}-${environment}-ai'
  logAnalytics: '${baseName}-${environment}-law'
}

// Storage Account for both Function App and blob monitoring
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: resourceNaming.storageAccount
  location: location
  sku: {
    name: storageSkuName
  }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    networkAcls: {
      defaultAction: 'Allow'
    }
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
  }

  // Blob service configuration
  resource blobService 'blobServices@2023-05-01' = {
    name: 'default'
    properties: {
      deleteRetentionPolicy: {
        enabled: true
        days: 7
      }
      containerDeleteRetentionPolicy: {
        enabled: true
        days: 7
      }
    }

    // Incoming container for new PNG files
    resource incomingContainer 'containers@2023-05-01' = {
      name: 'incoming'
      properties: {
        publicAccess: 'None'
        metadata: {
          purpose: 'Container for incoming PNG files to be processed'
        }
      }
    }

    // Archive container for processed PNG files
    resource archiveContainer 'containers@2023-05-01' = {
      name: 'archive'
      properties: {
        publicAccess: 'None'
        metadata: {
          purpose: 'Container for archived PNG files'
        }
      }
    }
  }
}

// Log Analytics Workspace for monitoring
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: resourceNaming.logAnalytics
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

// Application Insights for telemetry
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: resourceNaming.applicationInsights
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// App Service Plan for Function App
resource hostingPlan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: resourceNaming.hostingPlan
  location: location
  sku: {
    name: functionAppSkuName
    tier: functionAppSkuName == 'Y1' ? 'Dynamic' : 'ElasticPremium'
  }
  kind: 'functionapp'
  properties: {
    reserved: true // Linux hosting
  }
}

// Function App
resource functionApp 'Microsoft.Web/sites@2023-12-01' = {
  name: resourceNaming.functionApp
  location: location
  kind: 'functionapp,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: hostingPlan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'NODE|20'
      alwaysOn: functionAppSkuName != 'Y1'
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'WEBSITE_CONTENTSHARE'
          value: toLower(resourceNaming.functionApp)
        }
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'node'
        }
        {
          name: 'WEBSITE_NODE_DEFAULT_VERSION'
          value: '~20'
        }
        {
          name: 'APPINSIGHTS_INSTRUMENTATIONKEY'
          value: applicationInsights.properties.InstrumentationKey
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: applicationInsights.properties.ConnectionString
        }
        {
          name: 'AZURE_STORAGE_CONNECTION_STRING'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'INCOMING_CONTAINER_NAME'
          value: 'incoming'
        }
        {
          name: 'ARCHIVE_CONTAINER_NAME'
          value: 'archive'
        }
      ]
      ftpsState: 'FtpsOnly'
      minTlsVersion: '1.2'
    }
  }
}

// Outputs
@description('The name of the Function App')
output functionAppName string = functionApp.name

@description('The hostname of the Function App')
output functionAppHostName string = functionApp.properties.defaultHostName

@description('The name of the storage account')
output storageAccountName string = storageAccount.name

@description('The connection string for the storage account')
@secure()
output storageConnectionString string = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'

@description('The Application Insights connection string')
@secure()
output applicationInsightsConnectionString string = applicationInsights.properties.ConnectionString

@description('The resource group name')
output resourceGroupName string = resourceGroup().name
