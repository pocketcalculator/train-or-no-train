require('dotenv').config();
// Download the helper library from https://www.twilio.com/docs/node/install
// Find your Account SID and Auth Token at twilio.com/console
// and set the environment variables. See http://twil.io/secure
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;
const TWILIO_SOURCE_PHONE_NUMBER = process.env.TWILIO_SOURCE_PHONE_NUMBER
const DESTINATION_PHONE_NUMBER = process.env.DESTINATION_PHONE_NUMBER
const client = require('twilio')(accountSid, authToken);


client.calls
      .create({
         twiml: "<Response><Say voice=\"woman\">A train is currently blocking crossing number 7, 1, 8, 0, 7, 9, N, as in Nancy.  Once again, a A train is currently blocking crossing number 7, 1, 8, 0, 7, 9, N, as in Nancy.  Thank you for moving this train.</Say><Play>https://msft2025trainornotrain.blob.core.windows.net/web/gladys-knight-and-the-pips-midnight-train-to-georgia.mp3</Play></Response>",
         to: DESTINATION_PHONE_NUMBER,
         from: TWILIO_SOURCE_PHONE_NUMBER,
       })
      .then(call => console.log(call.sid));
