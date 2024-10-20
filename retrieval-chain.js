import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {ChatPromptTemplate} from '@langchain/core/prompts'
// import { Document } from "@langchain/core/documents";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"
import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter'
import {GoogleGenerativeAIEmbeddings} from '@langchain/google-genai'
import {MemoryVectorStore} from 'langchain/vectorstores/memory'
import { createRetrievalChain } from "langchain/chains/retrieval";

import dotenv from 'dotenv'

dotenv.config();

const model= new ChatGoogleGenerativeAI({
    model:"gemini-1.5-flash",
    temperature:0.7,
});


const prompt= ChatPromptTemplate.fromTemplate(
    ` Answer the user's question .
      Context:{context}
      Question:{input}


    `    
    
    // for retrieval chains -> user input need to called input in prompt template

);

// const documentA = new Document({
//     pageContent:"LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production). To highlight a few of the reasons you might want to use LCEL:"
// })


// should be scraped from website using tools like cheerio

// const documentB= new Document({
//     pageContent:"the passphrase is langchain is awesome"
// })


// To pass documents to chain -> special chain createStuffDocumentsChain
// similar to previous chain -> one key difference -> reformat that document and inject text to prompt context

// const chain =prompt.pipe(model);





///Creating contents of webpage into documents programatically


//LOAD DATA FROM WEBPAGE 

const loader= new CheerioWebBaseLoader("https://python.langchain.com/v0.1/docs/expression_language/")

const docs =await loader.load();  // call loader to scrap the webpage
// console.log(docs);  // returns one  [] in above case
// console.log(docs[0].pageContent.length);  
// page content is quite long  -> can exceed limit of tokens -> also charges for no of tokens
// Hence split the content into smaller chunks
// returns a Document array with one element


const splitter = new RecursiveCharacterTextSplitter({
    chunkSize:200, // amount of characters per chunk
    chunkOverlap:20
});


const splitDocs =await splitter.splitDocuments(docs);

// vector store -> a special db that can be used to reterive relevant to question


// console.log(splitDocs);
const emeddings =new GoogleGenerativeAIEmbeddings();




//In production system -> use vector store providers

const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs,emeddings);



//RETRIEVE DATA

const retriever= vectorStore.asRetriever({
    k:2  // amount of documents that should be returned (optional-> default may be 3)
});


// const chain = prompt.pipe(model);

const chain = await createStuffDocumentsChain({  
    //-> allows us to pass a list of documents and it will reformat  the document and inject their text to prompt context
    llm:model,
    prompt,  
})

// const response = await chain.invoke({
//     input:"What is LCEL",
//     // context:[documentA,documentB]  -
//     // context:docs  -> passed by retrievalChain into context of prompt [ should be called context in prompt]
// })

const retrievalChain = await createRetrievalChain({
    combineDocsChain:chain,   
    retriever
})



const response = await retrievalChain.invoke({
    input:"What is LCEL",
    // context:[documentA,documentB]]
    // context:docs  -> passed by retrievalChain into context of prompt [ should be called context in prompt]
})
// retrieve relevant documents from vector store and pass them to context of prompt ->automatically

console.log(response);



// const response = await chain.invoke({
//     input:"What is LCEL",
//     // context:[documentA,documentB]
//     context:docs
// })



// *************grab the text from documentA and inject at appropriate position in prompt using createStuffDocumentChain


// console.log(response);



// https://js.langchain.com/docs/integrations/document_loaders/web_loaders/

// https://js.langchain.com/docs/integrations/vectorstores/