import streamlit as st
import os

from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings

HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACE_TOKEN')

embedding=HuggingFaceEmbeddings()


'''
# pdf_directory = "data"
# pdf_files = [f for f in os.listdir(data_directory) if f.endswith('.pdf')]
loaders = []
for pdf_name in pdf_files:
    file_path = "{}/{}".format(pdf_directory, pdf_name )
    loader = PyPDFLoader(file_path)
    loaders.append(loader)
docs = []
for loader in loaders: 
    docs.extend(loader.load()) 
documents = text_splitter.split_documents(docs) 
docsearch = FAISS.from_documents(documents, embeddings)
'''
loader=UnstructuredPDFLoader('/cyber.pdf')


document=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=[" ", ",", "\n"])
docs=text_splitter.split_documents(document)
db=FAISS.from_documents(docs, embedding)

llm=HuggingFaceHub(repo_id="google/flan-t5-large",
                   model_kwargs={"temperature":0.9,
                                 "max_length":512})

question=st.text_input('Input question about Cyber Policy:')

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

chain=load_qa_chain(llm, chain_type="stuff", verbose=True)
query="What is ____ exclusion?"
docs=db.similarity_search(question)
response=chain.run(input_documents=docs, question=query)

st.write(response)