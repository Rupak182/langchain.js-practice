import * as dotenv from 'dotenv'
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { ChatPromptTemplate } from '@langchain/core/prompts';

import {ConversationChain} from 'langchain/chains'  // or llm chain
//Memory imports 

import {BufferMemory} from 'langchain/memory'   // use it to store the info in database
import { UpstashRedisChatMessageHistory } from '@langchain/community/stores/message/upstash_redis';
import { Runnable, RunnableSequence } from '@langchain/core/runnables'

dotenv.config();

const model = new ChatGoogleGenerativeAI({
    model:"gemini-1.5-flash",
    temperature:0.7
})


const prompt = ChatPromptTemplate.fromTemplate(
    `You are an ai assistant. 
    History:{history}
    Input:{input}`
);


const upstashChatHistory= new UpstashRedisChatMessageHistory({
    sessionId:"chat1",  //an unique identifier for the conversation -> use the same id for continuing conversation
    config:{
        url:process.env.UPSTASH_REDIS_REST_URL,
        token:process.env.UPSTASH_REDIS_REST_TOKEN
    }
})

const memory = new BufferMemory({   // use it to obtain message from conversation -> formatted to string -> injected to variable in prompt called history
    memoryKey:'history',   // can be called anything
    chatHistory:upstashChatHistory
})

//LCEL

//memory object does not have pipe method

//Using Chain classes

// const chain = new ConversationChain({
//     llm:model,
//     prompt,
//     memory
// })


//Using LCEL

// const chain= prompt.pipe(model)  

// Runnable sequence -> prompt is an extecutable function that does some processing after taking input and return output to next function in chain

// other way which also let us add our own function
const chain = RunnableSequence.from([
    {
        input:(initalInput)=> initalInput.input,  // pass to other next part of chain ,input->property is passed when chain is invoked
        memory:()=> memory.loadMemoryVariables()  // custom propery passed -> no input taken 
    },
    {
        input:(previousOutput)=> previousOutput.input,  //maybe be using output of previous output as input (specifically input)
        history:(previousOutput)=> previousOutput.memory,
    },
    prompt,  // previous one as memory is not expected by prompt, it needs input and history
    model
])


// const chain =prompt.pipe(model);

//Get responses

//Runnable approach does not update buffer memory automatically


// console.log(await memory.loadMemoryVariables())
// const input1 = {
//     input:"The passphrase is WORLD"
// }

// const response1 = await chain.invoke(input1);

// console.log(response1);

// await memory.saveContext(input1, {
//     output:response1.content
// })

console.log("Updated history",await memory.loadMemoryVariables())
const input2 = {
    input:"What is the passphrase?"
}


const response2 = await chain.invoke(input2);

console.log(response2);

await memory.saveContext(input2, {
    output:response2.content
})
