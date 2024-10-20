import * as dotenv from 'dotenv'

import {ChatGoogleGenerativeAI} from '@langchain/google-genai'

dotenv.config();

const model = new ChatGoogleGenerativeAI({
    model:"gemini-1.5-flash",
    temperature:0.7,
    maxOutputTokens:1000,   // max tokend willing to spenf
    verbose:true
})  // ;lang


// const response =await model.invoke('Hello');


const response =await model.invoke("Write a poem about ai");  // poem in chuncks
console.log(response)

// const response =await model.batch(['Hello', "How are you?"]);

// const response =await model.stream("Write a poem about ai");  // poem in chuncks


// for await (const chunck of response){
//     console.log(chunck.content);
// }


// const response =await model.streamLog("Write a poem about ai");  // response in chuncks


// for await (const chunck of response){   // return all steps
//     console.log(chunck.content);
// }


// console.log(response);

