// More control over conversation and type of responses

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {ChatPromptTemplate} from "@langchain/core/prompts"
import dotenv from 'dotenv'

dotenv.config();




// const model =new ChatGoogleGenerativeAI({
//     model:"gemini-1.5-flash",
//     temperature:0.7,

// })

// AI app returning joke based on a word by user (no general conversation) -> prompt template



 /// two methods
//In this method we type a system message toinstruct model to behave a certain way




// const prompt = ChatPromptTemplate.fromTemplate('You are a comedian. Tell a joke based on the following word {input}')  

// // console.log(await prompt.format({input:"human"}));  //Human: You are a comedian. Tell a joke based on the following word human


// //Create chain -> combine model and prompt

// const chain = prompt.pipe(model);

// //Call chain

// const response =await chain.invoke({
//     input:"human"
// })

// console.log(response);


// method-2


const model =new ChatGoogleGenerativeAI({
    model:"gemini-1.5-flash",
    temperature:0.7,

})

//Takes an array of key value pairs -> allows to pass several messages
const prompt =ChatPromptTemplate.fromMessages([
    [
    "system",
    "You are a chef. Create a recipe based on a word provided by the user.",],
    ["human", "{word}"],
]
);

//System: Generate a joke based on a word provided by the user.
//Human: boy

// console.log(await prompt.format({input:"boy"}));  

//Create chain -> combine model and prompt

const chain = prompt.pipe(model);

// Call chain

const response =await chain.invoke({
    word:"boy"
})

console.log(response);
