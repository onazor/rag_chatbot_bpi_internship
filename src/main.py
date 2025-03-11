import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentType, initialize_agent
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain.tools import TavilySearchResults
from dotenv import load_dotenv
from langchain.docstore.document import Document
import pandas as pd
from PIL import Image
import pytesseract
from database import init_db, add_message, get_chat_history
import sqlite3
from langchain.memory import ConversationBufferMemory

load_dotenv(dotenv_path="/Users/jhonchristianrozano/Documents/Programming/Python Programming/rag-pdf-web-chatbot/.env") 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

app = Flask(__name__)   
app.secret_key = 'bpi_internship_rag_chatbot'
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

init_db()

llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)

persist_dir = "./chromadb_store"
vectorstore = Chroma(
    collection_name="sampleCollection",
    embedding_function=embedding_model,
    persist_directory=persist_dir
)

# process PDF
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    vectorstore.add_documents(documents=chunks)

# process excel files
def process_xlsx(xlsx_path):
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    def merge_row(row):
        return " ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])
    merged_rows = df.apply(merge_row, axis=1)
    merged_text = "\n".join(merged_rows)
    print("Merged Excel Text:", merged_text)
    documents = [Document(page_content=merged_text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    vectorstore.add_documents(documents=chunks)

# process parquet files
def process_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)
    def merge_row(row):
        return " ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])
    merged_rows = df.apply(merge_row, axis=1)
    merged_text = "\n".join(merged_rows)
    print("Merged Parquet Text:", merged_text)
    documents = [Document(page_content=merged_text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    vectorstore.add_documents(documents=chunks)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        question = request.form.get("question", "")
        file = request.files.get("file")

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            if filename.endswith(".pdf"):
                process_pdf(file_path)
            elif filename.endswith(".xlsx"):
                process_xlsx(file_path)
            elif filename.endswith(".parquet"):
                process_parquet(file_path)
            else:
                return jsonify({"error": "Unsupported file format"})

        if 'chat_history' not in session:
            session['chat_history'] = []

        # save user's new question to database
        add_message("user", question)
        chat_history = get_chat_history()

        # create conversation memory and load past messages from DB
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        for msg in chat_history:
            if msg[0] == "user":
                memory.chat_memory.add_user_message(msg[1])
            else:
                memory.chat_memory.add_ai_message(msg[1])

        # prepare retrieval tool and other tools
        retriever = vectorstore.as_retriever()
        retrieval_tool = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        tools = [
            Tool(
                name="Document_Retrieval",
                func=retrieval_tool.run,
                description="Retrieve answers from the content of uploaded files stored in the vector database."
            ),
            Tool(
                name="Web_Search",
                func=tavily_tool.invoke,
                description="Search the web for additional context."
            )
        ]

        # use a conversational agent with memory
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            memory=memory
        )

        response = agent_executor.run(question)
        # save bot response to database
        add_message("bot", response)
        return jsonify({"answer": response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing your request."})

# for resetting files and chats
@app.route("/reset", methods=["POST"])
def reset():
    try:
        session.clear()
        upload_folder = app.config["UPLOAD_FOLDER"]
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_history')
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while resetting the session."})

# chat history
@app.route("/chat_history", methods=["GET"])
def chat_history():
    chat_history = get_chat_history()
    return jsonify({"chat_history": [{"role": msg[0], "content": msg[1]} for msg in chat_history]})

if __name__ == "__main__":
    app.run(debug=True) 
