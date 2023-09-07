import os
import streamlit as st

import faiss
from langchain.chat_models import ChatOpenAI

from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message


HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACE_TOKEN')
OPENAI_API_KEY = os.getenv['OPENAI_API_KEY']

loader=UnstructuredPDFLoader('/src/data/jane_eyre.pdf')

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
document=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=[" ", ",", "\n"])
all_splits=text_splitter.split_documents(document)
embedding=HuggingFaceEmbeddings()
vectorstore=FAISS.from_documents(documents=all_splits, embedding=embedding)

# App layout
st.title('Jane Eyre Information Retreival')
query=[]
query=st.text_input('Input question below:')


# look into inputting chat history to prompt?


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\
Make sure to avoid using any unclear pronouns.If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use six sentences maximum.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

condense_question_chain = LLMChain(
    llm=llm,
    prompt=CONDENSE_QUESTION_PROMPT,
)
doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)

qa_chain = create_qa_with_sources_chain(llm)

final_qa_chain = StuffDocumentsChain(
    llm_chain=qa_chain,
    document_variable_name="context",
    document_prompt=doc_prompt,
)

qa = ConversationalRetrievalChain(
    question_generator=condense_question_chain,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    memory=memory,
    combine_docs_chain=final_qa_chain,
    return_source_documents=True,
    verbose=True

)

#chat_bot, qa = st.tabs({'Chat Bot', 'Q & A')


#with chat_bot:

def conversational_chat(query):
    
    result = qa({"question": query})

    st.session_state['history'].append(result['chat_history'])
    st.session_state['history'].append((query, result["answer"]))
    st.session_state['sources'].append(( query, result['source_documents']))

    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'sources' not in st.session_state:
    st.session_state['sources']=[]

if 'answer' not in st.session_state:
    st.session_state['answer'] = ["What do you want to know about Jane Eyre?"]

if 'context' not in st.session_state:
    st.session_state['context'] = []
    
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Ask a question here", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['context'].append(user_input)
        st.session_state['answer'].append(output)

if st.session_state['answer']:
    with response_container:
        for i in range(len(st.session_state['answer'])):
            message(st.session_state["context"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["answer"][i], key=str(i), avatar_style="thumbs")
st.title('Chat history:')
st.write(st.session_state['history'])

st.title('Sources:')
st.write(st.session_state['sources'])


#with qa:

    #result = qa({"question": query})

    #st.markdown('**Answer**:')
    #st.write(result['answer'])

    #st.write('**Sources consulted**:')
    #st.write(result['source_documents'])


    #with st.expander('View full chat history:'):
        #st.write(result['chat_history'])

