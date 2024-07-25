import weaviate
import streamlit as st
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
HUGGINGFACE_API_KEY= os.getenv('HUGGINGFACE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')




# Initialize Weaviate client and other components
WEAVIATE_URL = "https://itk9ojwtuat8lomvl78yw.c0.asia-southeast1.gcp.weaviate.cloud"

client = weaviate.Client(
    url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

loader = PyPDFLoader("D:\SUMMER_OF_CODE\Final_project\FINAL_SCRAPED_DATA.pdf", extract_images=True)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
docs = text_splitter.split_documents(pages)

vector_db = Weaviate.from_documents(docs, embeddings, client=client, by_text=False)
retriever = vector_db.as_retriever()

template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatGroq(
    temperature=0.5,
    model="llama3-70b-8192",
    api_key="GROQ_API_KEY"
)

output_parser = StrOutputParser()

# Initialize conversation memory
conversation_memory = ConversationBufferMemory()

# Define RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)
def get_answer(question):
  return rag_chain.invoke(question)


