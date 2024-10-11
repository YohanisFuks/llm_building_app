import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain   
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain import SerpAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import Tool, tool, BaseTool   
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain import PromptTemplate, LLMChain
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory


load_dotenv()
st.set_page_config(page_title="Globebotter", page_icon="🌍", layout="wide")
st.header("Globebotter")

def display_msg(msg, author):
    st.session_state.messages.append(({'role': author, 'content': msg}))
    st.chat_message(author).write(msg)

#setup das tools
## search
search = SerpAPIWrapper()

## RAG
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200
)
raw_documents = PyPDFLoader('italy_travel.pdf').load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

#consolidação
tools = [
    Tool.from_function(
        func = search.run,
        name = 'search',
        description = 'searchs the internet for fresh and updated information not available at the documents'
    ),
    create_retriever_tool(
        db.as_retriever(),
        name = 'italy_planner',
        description = 'search and returns documents regarding italy places, hotels and restaurants'
    )
]


#memoria
memory = ConversationBufferMemory(
    return_messages = True,
    memory_key = 'chat_history',
    output_key = 'output'
)

#llm
llm = ChatOpenAI()


#criação do agente
agent = create_conversational_retrieval_agent(llm, tools, memory_key = 'chat_history', verbose = True)


user_query = st.text_input("** where are you plannning your next vacation?",
                           placeholder = 'ask me anything')


if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant',
                                     'content': 'how can I help you?'}]
    
if 'memory' not in st.session_state:
    st.session_state['memory'] = memory
    
    
for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])
    
    
if user_query:
    display_msg(user_query, 'user')
    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent(user_query, callbacks = [st_cb])
        st.session_state.messages.append({'role': 'assitant',
                                          'content': response})
        st.write(response)
        
        
        
if st.sidebar.button('reset chat history'):
    st.session_state.messages = []