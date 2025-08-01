# app.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import openai
import json
from typing import List
from datasets import load_dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configurar clave OpenAI desde variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializar FastAPI
app = FastAPI()

# Permitir CORS desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada para el bot
class Question(BaseModel):
    question: str

# Cargar corpus
print("Cargando corpus...")
dataset = load_dataset("json", data_files="corpus.jsonl")
docs = [d["text"] for d in dataset["train"]]

# Dividir texto en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.create_documents(docs)

# Embeddings y base vectorial
print("Generando vectorstore...")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Prompt personalizado
prompt_template = PromptTemplate(
    template="""
Respondé con precisión y sin inventar. Si no está en el corpus, ofrecé escribir a info@goethe.edu.ar.

Pregunta: {question}
Respuesta:
""",
    input_variables=["question"],
)

# Crear el chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True,
)

@app.post("/chat")
async def chat_api(q: Question):
    response = qa_chain(q.question)
    return {
        "answer": response["result"],
        "sources": [doc.metadata for doc in response["source_documents"]],
    }

@app.get("/")
def root():
    return {"message": "Bot Goethe AI activo"}