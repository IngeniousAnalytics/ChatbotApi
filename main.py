from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from fastapi.responses import PlainTextResponse
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage  # Correct imports
import os
from typing import List
from os import listdir
from os.path import isfile, join
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(filename='audit_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Global state
vectorstore = None
llm = None

# Initialize Google API key from the environment
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)  # Set the Google API key here
else:
    raise ValueError("Google API key not found in the .env file")

async def process_pdfs():
    global vectorstore, llm

    # Get a list of PDF files from the 'docs' folder
    pdf_docs_folder = "docs"
    pdf_docs = [join(pdf_docs_folder, f) for f in listdir(pdf_docs_folder) if isfile(join(pdf_docs_folder, f)) and f.lower().endswith(".pdf")]

    # Process PDF files
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)

    # Initialize Google Generative AI (Gemini) using Langchain's ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # Specify the model name

    print("PDFs processed successfully")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await process_pdfs()
    yield

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to the actual origin URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Use GoogleGenerativeAIEmbeddings instead of OpenAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Google embeddings
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

@app.post("/ask-question/")
async def ask_question(request: Request, question: str = Form(...)):
    if not vectorstore or not llm:
        return {"error": "No vectorstore or LLM available. Please upload and process PDFs first."}

    # Get the client IP address
    client_ip = request.client.host

    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question)
    if not docs:
        log_message = f"IP: {client_ip}, Question: {question}, Response: No relevant information found in the documents."
        logging.info(log_message)
        return {"response": "No relevant information found in the documents.", "chat_history": []}

    # Generate context from retrieved documents
    context = " ".join([doc.page_content for doc in docs])
    if not context:
        log_message = f"IP: {client_ip}, Question: {question}, Response: No relevant context found."
        logging.info(log_message)
        return {"response": "No relevant context found in the documents.", "chat_history": []}

    # Create prompt with context and question
    prompt = f"Based on the following context from the documents, answer the question strictly based on the context provided:\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    # Create message objects
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]

    # Generate response using Google Generative AI (Gemini) via Langchain
    try:
        response = await llm.agenerate([messages])  # Pass the formatted messages
        answer = response.generations[0][0].text.strip()  # Use .generations to access the text
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return {"error": "An error occurred while generating the response. Please try again later."}

    # Log the question and response along with the IP address
    log_message = f"IP: {client_ip}, Question: {question}, Response: {answer}"
    logging.info(log_message)

    return {"response": answer, "chat_history": []}

@app.get("/banned-words", response_class=PlainTextResponse)
async def get_banned_words():
    with open("docs/banned_words.txt", "r") as file:
        return file.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
