from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = GoogleGenerativeAI(model="models/text-bison-001", temperature=0.1)


def pdf_read(pdfs:list) -> str:
    raw_text = ''
    for i in pdfs:
        pdf = PdfReader(i)
        for i, page in enumerate(pdf.pages):
            raw_text += page.extract_text()
    return raw_text

def get_chunks(text:str) -> list:
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectordb(chunks:list):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectordb

def get_chain(vectordb):
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory
    )
    return chain