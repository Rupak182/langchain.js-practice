import * as dotenv from 'dotenv'
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { ChatPromptTemplate,MessagesPlaceholder } from '@langchain/core/prompts';
import {createToolCallingAgent, AgentExecutor} from 'langchain/agents'
import {TavilySearchResults} from '@langchain/community/tools/tavily_search'
import readline, { createInterface } from 'readline';
import {HumanMessage,AIMessage} from "@langchain/core/messages"  // schemas
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { createRetrieverTool } from 'langchain/tools/retriever'; 
dotenv.config();

const loader= new CheerioWebBaseLoader("https://python.langchain.com/v0.1/docs/expression_language/")
const docs =await loader.load(); 

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize:200, // amount of characters per chunk
    chunkOverlap:20
});

const splitDocs =await splitter.splitDocuments(docs);

const emeddings =new GoogleGenerativeAIEmbeddings();

const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs,emeddings);

const retriever= vectorStore.asRetriever({
    k:2  
});




const model = new ChatGoogleGenerativeAI({
    model: "gemini-1.5-flash",
    temperature: 0.7,
})

const prompt =ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant that answers the following questions as best as you can.You have access to the following tools"],
    ["human","{input}"],
    // ["placeholder","{agent_scratchpad}"]
    new MessagesPlaceholder("agent_scratchpad"),  // used by agent to keep track of steps maybe
    new MessagesPlaceholder("chat_history")
])

// addtional varaible to be included in prompt template -> done by agent to keep track of all the tools and steps (behind the scenes)
const searchTool= new TavilySearchResults();
const reteriverTool = createRetrieverTool(retriever,{
    name:"lcel_search",
    description:
    "Use this tool when searching for information about Lanchain Expression Language (LCEL)",  // for telling agent when to use the tool
})
const tools=[searchTool,reteriverTool]



//agent instead of chain
const agent=  createToolCallingAgent({
    llm:model,
    prompt,
    tools,   /// need atleast one tool
})


const agentExecutor = new AgentExecutor({
    agent,
    tools,
})




//Get user input
const rl = createInterface({
    input:process.stdin,
    output:process.stout
})


const chatHistory=[];

const askQuestion=()=>{

    rl.question("User: ",async (input)=>{
        if(input.toLowerCase=="exit"){
            r1.close();
            return;
        }

        const response =await agentExecutor.invoke({
            input:input,
            chat_history:chatHistory
        })
         
        console.log("Agent: " ,response.output);
        chatHistory.push(new HumanMessage(input))  
        chatHistory.push(new AIMessage(response.output))
        askQuestion();

    })
    
}

askQuestion();






// https://js.langchain.com/v0.1/docs/modules/agents/agent_types/