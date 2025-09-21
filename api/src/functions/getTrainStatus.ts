import { app, HttpRequest, HttpResponseInit, InvocationContext } from "@azure/functions";
import { CosmosClient } from "@azure/cosmos";

const endpoint = process.env.COSMOS_DB_CONNECTION_STRING;
const client = new CosmosClient(endpoint);
const databaseId = "devicesdb";
const containerId = "devices";

export async function getTrainStatus(request: HttpRequest, context: InvocationContext): Promise<HttpResponseInit> {
    context.log(`Http function processed request for url "${request.url}"`);

    try {
        const { resources: items } = await client
            .database(databaseId)
            .container(containerId)
            .items.query({
                query: "SELECT TOP 1 c.train, c.messageTimestamp FROM c ORDER BY c._ts DESC"
            })
            .fetchAll();

        if (items.length > 0) {
            const latestItem = items[0];
            
            // Robust date handling
            let dateObject;
            const ts = latestItem.messageTimestamp;
            context.log(`Raw timestamp from DB: ${ts}`);
            if (typeof ts === 'number') {
                // If the timestamp is in seconds, convert to milliseconds
                dateObject = new Date(ts * 1000);
            } else {
                // Otherwise, try to parse it directly
                dateObject = new Date(ts);
            }

            // Check if the date is valid before sending
            if (isNaN(dateObject.getTime())) {
                context.log(`Failed to parse date: ${ts}`);
                return { status: 500, body: "Invalid date format in database." };
            }

            return {
                jsonBody: {
                    train: latestItem.train,
                    timestamp: dateObject.toISOString() // Send in standard ISO format
                }
            };
        } else {
            return {
                status: 404,
                body: "No items found in the container."
            };
        }
    } catch (error) {
        context.log(`Error fetching from Cosmos DB: ${error.message}`);
        return {
            status: 500,
            body: "Error connecting to or fetching from the database."
        };
    }
}

app.http('getTrainStatus', {
    methods: ['GET'],
    authLevel: 'anonymous',
    handler: getTrainStatus
});
