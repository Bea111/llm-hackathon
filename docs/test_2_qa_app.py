import os
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings


HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACE_TOKEN')

embedding=HuggingFaceEmbeddings()

loader=UnstructuredPDFLoader('/jane_eyre.pdf')

document=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=[" ", ",", "\n"])
docs=text_splitter.split_documents(document)
vectorstore=FAISS.from_documents(docs, embedding)



llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


doc_prompt_temp_1 = PromptTemplate(
    template= """"Use the following most relevant piece of context to answer the question at the end. Mainly use the first piece of context. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer.
    {page_content}\nSource: {source}""", # look at the prompt does have page#
    input_variables=["page_content", "source"] )



source_qa_chain = create_qa_with_sources_chain(llm)


final_qa_chain = StuffDocumentsChain(
    llm_chain=source_qa_chain, 
    document_variable_name='context',
    document_prompt=doc_prompt_temp_1,
)

qa_chain = RetrievalQA(retriever=vectorstore.as_retriever(),
                                       combine_documents_chain=final_qa_chain,
                                       return_source_documents=True,                                         
                                       verbose=True)

query = "How did Jane wear her hair?"

result = qa_chain({"query": query})

print(result['result'])
print(len(result['source_documents']))
result['source_documents'][0]
#result #returns all results

