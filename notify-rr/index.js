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
         url: 'http://demo.twilio.com/docs/voice.xml',
         to: DESTINATION_PHONE_NUMBER,
         from: TWILIO_SOURCE_PHONE_NUMBER,
       })
      .then(call => console.log(call.sid));
