import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {ChatPromptTemplate} from '@langchain/core/prompts'
// import { Document } from "@langchain/core/documents";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"
import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter'
import {GoogleGenerativeAIEmbeddings} from '@langchain/google-genai'
import {MemoryVectorStore} from 'langchain/vectorstores/memory'
import { createRetrievalChain } from "langchain/chains/retrieval";
import {AIMessage,HumanMessage} from '@langchain/core/messages'
import { MessagesPlaceholder} from '@langchain/core/prompts'
import {createHistoryAwareRetriever} from 'langchain/chains/history_aware_retriever'

import dotenv from 'dotenv'

dotenv.config();


//LOAD DATA FROM WEBPAGE 


const createVectorStore = async()=>{
    const loader= new CheerioWebBaseLoader("https://python.langchain.com/v0.1/docs/expression_language/")

    const docs =await loader.load();  // call loader to scrap the webpage
    
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize:200, // amount of characters per chunk
        chunkOverlap:20
    });
    
    
    const splitDocs =await splitter.splitDocuments(docs);
    
    const emeddings =new GoogleGenerativeAIEmbeddings();
    
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs,emeddings);

    return vectorStore
    
}


const createChain = async(vectorStore)=>{
    const model= new ChatGoogleGenerativeAI({
        model:"gemini-1.5-flash",
        temperature:0.7,
    });
    
    
    const prompt= ChatPromptTemplate.fromMessages([
        ["system","Answer the user's question based on the following context:{context}"],

        new MessagesPlaceholder("chat_history"), // Accept array of messages and convert to string
        ["user","{input}"]
        

    
    ]);

        
    const chain = await createStuffDocumentsChain({  
        //-> allows us to pass a list of documents and it will reformat  the document and inject their text to prompt context
        llm:model,
        prompt,  
    })

    const retriever= vectorStore.asRetriever({
        k:2  // amount of documents that should be returned (optional-> default may be 3)
    });
    // Does not allow to pass chat_history to get more relevant documents from vector store (only input passed)


    const retrieverPrompt = ChatPromptTemplate.fromMessages([
        new  MessagesPlaceholder("chat_history"),
        ["user","{input}"],
        ["user","Given the above conversation , generate a search query to lookup in order to get relevant information for the conversation"]
    ])


    const historyAwareRetriever= await createHistoryAwareRetriever({
        llm:model,
        retriever,
        rephrasePrompt:retrieverPrompt
    })

    
    const conversationChain = await createRetrievalChain({
        combineDocsChain:chain,   
        retriever:historyAwareRetriever

    })

    return conversationChain
}

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);



const chatHistory= [
    new HumanMessage("Hello"),
    new AIMessage("Hi, how can i help you"),
    new HumanMessage("What is LCEL?"),
    new AIMessage("LCEL stands for Langchain Expresssion Language"),
]


const response = await chain.invoke({
    input:"What is it",
    chat_history:chatHistory// expecting text instead of array
})
// retrieve relevant documents from vector store and pass them to context of prompt ->automatically

console.log(response);


