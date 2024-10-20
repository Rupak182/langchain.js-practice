import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {ChatPromptTemplate} from "@langchain/core/prompts"
import dotenv from 'dotenv'
import { StringOutputParser,CommaSeparatedListOutputParser } from '@langchain/core/output_parsers'
import {StructuredOutputParser} from 'langchain/output_parsers'
import z from 'zod'

dotenv.config();


const model =new ChatGoogleGenerativeAI({
    model:"gemini-1.5-flash",
    temperature:0.7,

})

async function callStringOutputParser(){
    const prompt =ChatPromptTemplate.fromMessages([
        [
        "system",
        "You are a comedian. Create a short joke based on the following word ",],
        ["human", "{word}"],
    ]
    );
    
    //Create parser
    
    const parser = new StringOutputParser();
    
    const chain = prompt.pipe(model).pipe(parser);
    
    // Call chain
    
    return await chain.invoke({
        word:"boy"
    })
    
}


async function callListOutputParser(){
    const prompt = ChatPromptTemplate.fromTemplate("Provide 5 synonyms seperated by commas for the following word {word}");
    const outputParser= new CommaSeparatedListOutputParser();   // string to javascript array

    const chain = prompt.pipe(model).pipe(outputParser);

    return await chain.invoke({
        word:"happy"
    })

}


//Structured Ouput Parser
async function callStructuredParser(){
    const prompt = ChatPromptTemplate.fromTemplate(`Extract information from the following phrase.
        Formatting instructions: {format_instructions}
        Phrase:{phrase}`);
    const outputParser= StructuredOutputParser.fromNamesAndDescriptions({
        name:"The name of the person",
        age:"The age of the person",
    });   // simple structure
    // the structure is converted to formatting instructions understandable by model

    const chain = prompt.pipe(model).pipe(outputParser);

    console.log(outputParser.getFormatInstructions())

    return await chain.invoke({
        phrase:"Max is 31 years old",
        format_instructions:outputParser.getFormatInstructions()
    })

}

async function callZodOuputParser(){
    const prompt =ChatPromptTemplate.fromTemplate(`
        Extract information from the following phrase.
        Formatting instructions: {format_instructions}
        Phrase:{phrase}`);

    const outputParser =StructuredOutputParser.fromZodSchema(
        z.object({
            recipe:z.string().describe("name for recipe"),  // for model to understand meaning of field add describe
            ingredients:z.array(z.string()).describe("Ingredients")
        })
    )
    const chain = prompt.pipe(model).pipe(outputParser);
    // console.log(outputParser.getFormatInstructions())

    return await chain.invoke({
        phrase:"The ingredients for a Spaghetti  recipe are tomatoes, garlic, herbs",
        format_instructions:outputParser.getFormatInstructions()
    })

    
}




// const response =await callStringOutputParser();
// const response = await callListOutputParser();
// const response = await callStructuredParser();
const response =  await callZodOuputParser()
console.log(response);



// https://js.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/