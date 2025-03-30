import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from openai import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain.docstore.document import Document
import pandas as pd
import sqlite3
from PIL import Image
import pytesseract
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import uuid
from datetime import datetime

# Load environment variables
load_dotenv(dotenv_path="/Users/jhonchristianrozano/Documents/Programming/Python Programming/bpi-business-banking-bria/.env")

app = Flask(__name__)
app.secret_key = 'chatgpt_mvp_secret_key'
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# For industry classification
app.config["SAMPLE_DATA_FOLDER"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
os.makedirs(app.config["SAMPLE_DATA_FOLDER"], exist_ok=True)

# Create sample files if they don't exist
def ensure_sample_files_exist():
    # Retail sample
    retail_file = os.path.join(app.config["SAMPLE_DATA_FOLDER"], "retail_transactions.csv")
    if not os.path.exists(retail_file):
        with open(retail_file, 'w') as f:
            f.write("""date,transaction_id,description,category,amount,vendor,payment_method
                    2023-01-15,TX100001,Inventory purchase - Winter clothing,Inventory,5420.50,StyleFactory Wholesale,Bank Transfer
                    2023-01-20,TX100002,Point of sale system monthly fee,Technology,129.99,RetailTech Solutions,Credit Card
                    2023-01-22,TX100003,Shelf display units purchase,Store Equipment,1250.00,ShopFittings Inc.,Bank Transfer
                    2023-01-31,TX100006,Sales - Clothing department,Sales Revenue,12540.75,Various Customers,Mixed
                    2023-01-31,TX100007,Sales - Accessories department,Sales Revenue,4320.25,Various Customers,Mixed
                    2023-02-01,TX100008,Employee wages - 5 sales associates,Payroll,8500.00,Staff Payroll,Bank Transfer
                    2023-02-10,TX100012,Inventory purchase - Spring collection,Inventory,7650.25,StyleFactory Wholesale,Bank Transfer
                    2023-02-18,TX100015,Sales - Online store,Sales Revenue,3245.65,Various Customers,Online Payment""")

    # Restaurant sample
    restaurant_file = os.path.join(app.config["SAMPLE_DATA_FOLDER"], "restaurant_transactions.csv")
    if not os.path.exists(restaurant_file):
        with open(restaurant_file, 'w') as f:
            f.write("""date,transaction_id,description,category,amount,vendor,payment_method
                    2023-01-03,TX200001,Food inventory purchase - Produce,Food Inventory,1420.75,FreshFarms Suppliers,Bank Transfer
                    2023-01-05,TX200002,Food inventory purchase - Meat,Food Inventory,2850.50,Quality Meats Inc.,Bank Transfer
                    2023-01-07,TX200003,Beverage inventory purchase,Beverage Inventory,1875.25,Beverage Wholesale Co.,Bank Transfer
                    2023-01-15,TX200008,Dinner service sales,Food Sales,3450.80,Various Customers,Mixed
                    2023-01-15,TX200009,Beverage sales,Beverage Sales,1250.40,Various Customers,Mixed
                    2023-01-16,TX200010,Lunch service sales,Food Sales,2180.60,Various Customers,Mixed
                    2023-01-28,TX200021,Food delivery platform commission,Delivery Service,320.50,UberEats,Direct Debit""")
            
    # Tech company sample
    tech_file = os.path.join(app.config["SAMPLE_DATA_FOLDER"], "tech_company_transactions.csv")
    if not os.path.exists(tech_file):
        with open(tech_file, 'w') as f:
            f.write("""date,transaction_id,description,category,amount,vendor,payment_method
                    2023-01-04,TX300001,Cloud server hosting fees,Cloud Infrastructure,4520.75,AWS,Credit Card
                    2023-01-05,TX300002,Software development tools licenses,Software Licenses,2150.00,JetBrains,Credit Card
                    2023-01-07,TX300003,Office rental - Tech Park,Rent,5800.00,TechSpace Properties,Bank Transfer
                    2023-01-15,TX300008,SaaS subscription revenue,Revenue,28750.50,Various Customers,Online Payments
                    2023-01-18,TX300009,API usage revenue,Revenue,12450.25,Various Customers,Online Payments
                    2023-01-20,TX300010,Database service subscription,Cloud Infrastructure,1850.30,MongoDB Atlas,Credit Card
                    2023-01-25,TX300012,Professional development courses,Training,3500.00,Udemy for Business,Credit Card""")

# Ensure sample files exist when app starts
ensure_sample_files_exist()

# Database setup
def init_db():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def add_message(role, content):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO chat_history (role, content) VALUES (?, ?)', (role, content))
    conn.commit()
    conn.close()

def get_chat_history():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT role, content FROM chat_history ORDER BY id')
    history = cursor.fetchall()
    conn.close()
    return history

init_db()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialize OpenAI client and LLMs
client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
instructions_llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Initialize embedding model and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
persist_dir = "./chromadb_store"
vectorstore = Chroma(
    collection_name="userDocuments",
    embedding_function=embedding_model,
    persist_directory=persist_dir
)

# Initialize Tavily search tool
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY)

# Functions for processing different file types
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    vectorstore.add_documents(documents=chunks)
    return f"Processed PDF: {os.path.basename(pdf_path)}"

def process_xlsx(xlsx_path):
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    def merge_row(row):
        return " ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])
    merged_rows = df.apply(merge_row, axis=1)
    merged_text = "\n".join(merged_rows)
    documents = [Document(page_content=merged_text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    vectorstore.add_documents(documents=chunks)
    return f"Processed Excel: {os.path.basename(xlsx_path)}"

def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    def merge_row(row):
        return " ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])
    merged_rows = df.apply(merge_row, axis=1)
    merged_text = "\n".join(merged_rows)
    documents = [Document(page_content=merged_text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    vectorstore.add_documents(documents=chunks)
    return f"Processed CSV: {os.path.basename(csv_path)}"

def process_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)
    def merge_row(row):
        return " ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])
    merged_rows = df.apply(merge_row, axis=1)
    merged_text = "\n".join(merged_rows)
    documents = [Document(page_content=merged_text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    vectorstore.add_documents(documents=chunks)
    return f"Processed Parquet: {os.path.basename(parquet_path)}"

def process_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    doc = Document(page_content=text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents([doc])
    vectorstore.add_documents(documents=chunks)
    return f"Processed Image: {os.path.basename(image_path)}"

# Industry Classification function
def classify_industry(transaction_data_path):
    try:
        print(f"Processing file: {transaction_data_path}")
        # Read the file based on extension
        file_ext = os.path.splitext(transaction_data_path)[1].lower()
        
        if file_ext == '.csv':
            import pandas as pd
            df = pd.read_csv(transaction_data_path)
        elif file_ext in ['.xls', '.xlsx']:
            import pandas as pd
            df = pd.read_excel(transaction_data_path)
        elif file_ext == '.parquet':
            import pandas as pd
            df = pd.read_parquet(transaction_data_path)
        else:
            return {"error": "Unsupported file format. Please upload CSV, Excel, or Parquet files."}
            
        # Convert dataframe to string for analysis
        transaction_summary = df.head(50).to_string(index=False)
        
        # Create a more detailed prompt for the analysis
        prompt = f"""
        Based on the following transaction data, classify the business into an appropriate industry category.
        
        Transaction Data Sample:
        {transaction_summary}
        
        Analyze the transaction patterns, vendor types, expense categories, and revenue streams to determine the most appropriate industry classification.
        
        Provide a detailed, well-formatted analysis with these sections:
        
        ## Primary Industry Category
        [Identify the main industry sector like Retail, Manufacturing, Technology, etc.]
        
        ## Sub-Industry Classification
        [Provide a more granular classification within the primary industry]
        
        ## Confidence Level
        [High/Medium/Low with explanation of why]
        
        ## Key Transaction Indicators
        - [List 3-5 specific transaction patterns or data points that led to this classification]
        - [Include specific evidence from the data]
        
        ## Alternative Classifications
        [List 1-2 other potential industries this could belong to with brief explanations]
        
        ## Recommended Financial Products
        [Based on the industry classification, suggest 2-3 specific financial products or services that would be relevant]
        
        Format your response with clear headings, bullet points, and professional language.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst specializing in business classification. You analyze transaction data to help banks provide targeted financial services."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        classification_text = response.choices[0].message.content
        
        # Return both success and the detailed classification
        return {
            "success": True,
            "classification": classification_text,
            "message": "Industry classification completed successfully."
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Error classifying industry: {str(e)}"}

# Agent tools
def generate_tavily_instructions(query: str) -> str:
    prompt = (
        f'You are a research assistant. The user asked: "{query}"\n'
        "Please provide a short set of instructions or keywords to guide a web search.\n"
        "Do NOT provide a final answer. Only produce instructions or keywords relevant to the user query."
    )
    instructions = instructions_llm.invoke(prompt)
    if isinstance(instructions, dict) and "text" in instructions:
        return instructions["text"].strip()
    elif isinstance(instructions, str):
        return instructions.strip()
    else:
        return str(instructions).strip()

def process_link_content(link_info: str) -> str:
    prompt = f"""
    You are a research assistant. Summarize the following web search results by extracting key details:
    {link_info}
    Provide your answer as plain text.
    """
    result = instructions_llm.invoke(prompt)
    if isinstance(result, dict) and "text" in result:
        output = result["text"].strip()
    elif isinstance(result, str):
        output = result.strip()
    else:
        output = str(result).strip()
    return " ".join(output.split())

def safe_tavily_search(query: str) -> str:
    try:
        result = tavily_tool.invoke(query)
        if not isinstance(result, str):
            result = str(result)
        return result.strip()
    except Exception as e:
        return f"Web search failed: {str(e)}. Using local data only."

def list_uploaded_files():
    files = []
    upload_folder = app.config["UPLOAD_FOLDER"]
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        if os.path.isfile(file_path):
            files.append(filename)
    return files

# Routes
@app.route("/")
def index():
    # Check if session exists, if not initialize it
    if 'messages' not in session:
        # First check if there are messages in the database
        chat_history = get_chat_history()
        if chat_history:
            # If there's existing history, load it into the session
            session['messages'] = [
                {"role": "system", "content": "You are Bria, a specialized business banking assistant for BPI. You should ONLY discuss topics related to business banking, finance, and your specific capabilities. For any other topics outside this scope, politely explain that you cannot provide assistance."}
            ]
            for role, content in chat_history:
                session['messages'].append({"role": role, "content": content})
        else:
            # If no history, initialize with system message
            session['messages'] = [
                {"role": "system", "content": "You are Bria, a specialized business banking assistant for BPI. You should ONLY discuss topics related to business banking, finance, and your specific capabilities. For any other topics outside this scope, politely explain that you cannot provide assistance."}
            ]
    
    return render_template("index.html", uploaded_files=list_uploaded_files())

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        result = "File uploaded but not processed"
        if filename.lower().endswith(".pdf"):
            result = process_pdf(file_path)
        elif filename.lower().endswith(".xlsx"):
            result = process_xlsx(file_path)
        elif filename.lower().endswith(".csv"):
            result = process_csv(file_path)
        elif filename.lower().endswith(".parquet"):
            result = process_parquet(file_path)
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            result = process_image(file_path)
        else:
            return jsonify({"error": "Unsupported file format"})
        
        return jsonify({"success": True, "message": result, "files": list_uploaded_files()})
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "")
        print(f"Processing user message: '{user_message}'")
        
        # Skip the relevance check entirely - process all messages
        
        # Add user message to session history and database
        session['messages'].append({"role": "user", "content": user_message})
        add_message("user", user_message)
        
        # Create conversation memory and load past messages from DB
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chat_history = get_chat_history()
        for msg in chat_history:
            if msg[0] == "user":
                memory.chat_memory.add_user_message(msg[1])
            else:
                memory.chat_memory.add_ai_message(msg[1])
        
        # Create a retriever with improved search parameters
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": 8, "fetch_k": 20}  # Fetch more candidates and return more results
        )

        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True  # Keep this True for debugging
        )

        # Create wrapper function that handles the dictionary return format
        def retrieval_tool_func(query: str) -> str:
            """Extract information from documents in the vector store."""
            try:
                # The chain returns a dict with 'result' and 'source_documents' keys
                chain_result = qa_chain({"query": query})
                
                # Extract just the result text for the Tool
                if isinstance(chain_result, dict) and "result" in chain_result:
                    return chain_result["result"]
                else:
                    return str(chain_result)
            except Exception as e:
                return f"Error searching documents: {str(e)}. Please try a different query."
        
        # set up tools for the agent
        tools = [
            Tool(
                name="Document_Retrieval",
                func=retrieval_tool_func,  # Use our custom wrapper
                description="Retrieve answers from local database of uploaded files. Use this for any questions about uploaded documents or company information."
            ),
            Tool(
                name="Generate_Tavily_Instructions",
                func=generate_tavily_instructions,
                description="Generate instructions or keywords for web search."
            ),
            Tool(
                name="Web_Search",
                func=safe_tavily_search,
                description="Search the web for additional context using instructions or keywords."
            ),
            Tool(
                name="Process_Link_Content",
                func=process_link_content,
                description="Process web search results to extract key details."
            )
        ]
        
        # set up agent with specific workflow instructions
        prefix = """You are a helpful assistant specializing in banking and finance, particularly for Business Banking at BPI. You work with both documents and web searches.

        VERY IMPORTANT: You can answer questions related to:
        1. Any types of businesses and their operations
        2. Business banking services and products
        3. Finance and financial advice
        4. Information about specific industries or companies
        5. Local businesses and market trends

        This includes providing information about food businesses, trendy establishments, seasonal business trends, and other business-related topics which are relevant for understanding potential client needs.

        For every user query, follow these steps:
        1. Use "Document_Retrieval" to search the local database.
        2. If relevant documents are found, analyze them for your response.
        3. Use "Web_Search" to gather additional context.
        4. Combine all sources into one final, comprehensive answer.

        IMPORTANT: Make sure your responses are compatible with the output parser. Always follow this format:
        Thought: <your reasoning>
        Action: <tool_name>
        Action Input: <tool_input>

        After receiving the tool's observation:
        Thought: <your reasoning>

        When you have enough information for a response:
        Thought: I now have enough information to answer the query.
        AI: <your final answer>

        SPECIALIZED BUSINESS FUNCTIONS:

        For Report Generation queries:
        - Focus on summarizing loan portfolio data, credit utilization metrics, and campaign performance
        - Structure your response with clear sections for each metric type
        - Highlight KPIs and trends
        - Suggest actionable insights based on the data

        For Risk Assessment queries:
        - Focus on creditworthiness factors, default risk indicators, and high-value client traits
        - Provide methodical frameworks for evaluating client profiles
        - Explain predictive signals for potential loan defaults
        - Outline criteria for identifying promising leads for financial products

        For Query Handling requests:
        - Focus on loan eligibility criteria, application requirements, and campaign details
        - Structure responses as clear FAQ-style answers
        - Provide step-by-step guidance for handling client inquiries
        - Include examples of effective responses

        For Report Analysis queries:
        - Focus on extracted information from financial documents
        - Reference specific metrics and insights from generated reports
        - Provide contextual explanations of financial data
        - Connect insights to relevant banking products and services
        - Structure responses with clear sections for metrics, analysis, and recommendations

        ALWAYS FORMAT YOUR RESPONSES IN MARKDOWN:
        - Use **bold text** for all important information, key facts, names, figures, and highlights
        - Use ## for main headings (e.g., ## Forex Businesses in Makati)
        - Use ### for subheadings to organize information clearly
        - Format all links as [descriptive link text](URL) to make them clickable
        - Use bullet points (- item) for lists of related items
        - Use numbered lists (1. step) for sequential steps or ranked items
        - Use blockquotes (> text) for direct quotes or important notes
        - Use `code` formatting for technical terms, commands, or code snippets
        - Use tables for structured data when appropriate

        Make your response well-structured and use Markdown to improve readability and highlight the most important information. When displaying data, use clear formatting with headers, bullet points, and tables where appropriate. Always include direct, clickable links to sources when available.
        """
        
        # initialize agent with memory
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,  
            handle_parsing_errors=True,
            max_iterations=3,  
            early_stopping_method="generate", 
            memory=memory,
            agent_kwargs={"prefix": prefix}
        )
        
        # run agent to get response
        try:
            print(f"Processing query: {user_message}")
            response = agent_executor.run(user_message)
            print(f"Agent completed successfully")
        except ValueError as e:
            # handle parsing errors
            print(f"Agent parsing error: {e}")
            if "Could not parse LLM output" in str(e):
                # extract the content after "Could not parse LLM output: `" and before the last "`"
                error_message = str(e)
                start_marker = "Could not parse LLM output: `"
                end_marker = "`\nFor troubleshooting"
                
                if start_marker in error_message and end_marker in error_message:
                    start_idx = error_message.find(start_marker) + len(start_marker)
                    end_idx = error_message.find(end_marker)
                    response = error_message[start_idx:end_idx].strip()
                    print(f"Extracted direct response from parsing error")
                else:
                    # Default response if we can't extract
                    response = """
                    ## Response

                    I've analyzed your query and have some insights to share. However, due to the complexity of the information, I couldn't retrieve all specific details you might be looking for.

                    Please review the information below and let me know if you need further clarification on any particular aspect.
                    """
            else:
                # Handle other value errors
                response = """
                ## Information Request

                I encountered some challenges while processing your query. Here are some general insights that might help:

                - **Banking Services**: BPI offers comprehensive banking solutions for businesses of all sizes
                - **Financial Analysis**: For detailed financial analysis, we recommend submitting specific financial documents
                - **Further Assistance**: Our banking specialists can provide personalized guidance for your specific situation

                How else can I assist you today?
                """
        except Exception as e:
            # Fallback for other exceptions
            print(f"Agent error: {e}")
            response = """
            ## Response Error

            I encountered an issue while processing your request. Let me provide you with some general information:

            ### General Banking Information

            - **Banking Services**: Our banking services include loans, credit assessments, and financial reporting
            - **Contact Support**: For more detailed information, please contact our support team
            - **Try Again**: You can try rephrasing your question or providing more specific details

            I apologize for the inconvenience. How else can I assist you today?
            """
        
        # Add assistant response to session history and database
        session['messages'].append({"role": "assistant", "content": response})
        add_message("bot", response)
        
        return jsonify({"message": response})
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    try:
        # Clear session messages (but maintain session) - UPDATED SYSTEM PROMPT
        session['messages'] = [
            {"role": "system", "content": "You are Bria, a helpful assistant for BPI. You can provide information on businesses, business banking, finance, specific businesses, and other business-related topics including market trends and industry information."}
        ]
        
        # Clear database chat history
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_history')
        conn.commit()
        conn.close()
        
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error resetting chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/files", methods=["GET"])
def get_files():
    return jsonify({"files": list_uploaded_files()})

@app.route("/delete_file/<filename>", methods=["POST"])
def delete_file(filename):
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(filename))
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"success": True, "files": list_uploaded_files()})
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/classify_industry", methods=["POST"])
def handle_industry_classification():
    try:
        # Check if a file was uploaded
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No selected file"})
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            
            # run industry classification
            result = classify_industry(file_path)
            
            # add to chat history
            if "success" in result and "classification" in result:
                add_message("bot", f"## Industry Classification Results\n\n{result['classification']}")
            
            return jsonify(result)
        
        # or check if user wants sample data
        elif request.json and request.json.get("useSampleData"):
            ensure_sample_files_exist()
            
            # use retail transactions as default sample
            sample_file_path = os.path.join(app.config["SAMPLE_DATA_FOLDER"], "retail_transactions.csv")
            # verify the file exists
            if not os.path.exists(sample_file_path):
                sample_file_path = os.path.join(app.config["SAMPLE_DATA_FOLDER"], "emergency_sample.csv")
                with open(sample_file_path, 'w') as f:
                    f.write("""date,transaction_id,description,category,amount,vendor,payment_method
                        2023-01-15,TX100001,Inventory purchase - Winter clothing,Inventory,5420.50,StyleFactory Wholesale,Bank Transfer
                        2023-01-31,TX100006,Sales - Clothing department,Sales Revenue,12540.75,Various Customers,Mixed
                        2023-02-01,TX100008,Employee wages - 5 sales associates,Payroll,8500.00,Staff Payroll,Bank Transfer""")
            
            # run classification on the sample file
            result = classify_industry(sample_file_path)
            
            # add to chat history
            if "success" in result and "classification" in result:
                add_message("bot", f"## Industry Classification Results\n\n{result['classification']}")
            
            return jsonify(result)
        else:
            return jsonify({"error": "No file uploaded or sample data requested"})
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route("/chat_history", methods=["GET"])
def get_history():
    chat_history = get_chat_history()
    return jsonify({"chat_history": [{"role": msg[0], "content": msg[1]} for msg in chat_history]})

# create reports directory with absolute path
app.config["REPORTS_FOLDER"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(app.config["REPORTS_FOLDER"], exist_ok=True)

# function to analyze financial document
def analyze_financial_document(file_path, settings):
    """
    Analyze PDF or Excel file to extract financial insights relevant for business banking
    """
    try:
        print(f"Analyzing file: {file_path}")
        # read the file based on extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # extract content based on file type
        if file_ext == '.pdf':
            # Process PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_content = " ".join([doc.page_content for doc in documents])
            
            # Also store in vector store for QA
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            vectorstore.add_documents(documents=chunks)
            
        elif file_ext in ['.xlsx', '.xls']:
            # Process Excel
            df = pd.read_excel(file_path)
            text_content = df.to_string()
            
            # Store in vector store for QA
            def merge_row(row):
                return " ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])
            merged_rows = df.apply(merge_row, axis=1)
            merged_text = "\n".join(merged_rows)
            documents = [Document(page_content=merged_text)]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            vectorstore.add_documents(documents=chunks)
            
        elif file_ext == '.csv':
            # Process CSV
            df = pd.read_csv(file_path)
            text_content = df.to_string()
            
            # Store in vector store for QA
            def merge_row(row):
                return " ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])
            merged_rows = df.apply(merge_row, axis=1)
            merged_text = "\n".join(merged_rows)
            documents = [Document(page_content=merged_text)]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            vectorstore.add_documents(documents=chunks)
            
        else:
            return {"error": "Unsupported file format. Please upload PDF, Excel, or CSV files."}
        
        # create a prompt for analysis based on report type
        report_type = settings.get('reportType', 'comprehensive')
        
        if report_type == 'loan-portfolio':
            analysis_focus = "Analyze the loan portfolio data focusing on outstanding balances, risk levels, interest rates, repayment performance, and portfolio diversity."
        elif report_type == 'financial-metrics':
            analysis_focus = "Extract and analyze key financial metrics including revenue, expenses, profitability ratios, liquidity ratios, and growth trends."
        elif report_type == 'campaign-performance':
            analysis_focus = "Evaluate marketing campaign performance including conversion rates, ROI, customer acquisition costs, reach, and engagement metrics."
        elif report_type == 'text-summary':
            # Simple text summarization option
            prompt = f"""
            Please provide a clear, concise summary of the following document:
            
            {text_content[:15000]}
            
            Focus on extracting the main points, key information, and overall message of the document.
            Format your response with clear headings and bullet points where appropriate.
            """
            
            # Send to LLM for summarization
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a document summarization specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            analysis_text = response.choices[0].message.content
            
            # Skip metrics for text summary and use placeholder data
            metrics = [
                {"name": "Document Length", "value": min(len(text_content), 15000), "unit": " chars"},
                {"name": "Sections", "value": text_content.count('#'), "unit": ""},
                {"name": "Key Points", "value": 5, "unit": ""},
                {"name": "Readability", "value": 8.5, "unit": "/10"}
            ]
            
            # Generate chart for visual
            chart_path = generate_metrics_chart(metrics)
            
            # Generate PDF report
            report_path = generate_pdf_report(
                analysis_text, 
                chart_path, 
                os.path.basename(file_path),
                settings
            )
            
            # Get the relative URL for the PDF
            report_url = f"/download_report/{os.path.basename(report_path)}"
            
            # Create a summary for the chat message
            summary = "I've summarized your document. Here's the key information contained within it."
            
            return {
                "success": True,
                "summary": summary,
                "details": analysis_text,
                "reportUrl": report_url,
                "metrics": metrics
            }
        else:  # comprehensive
            analysis_focus = "Provide a comprehensive analysis covering financial performance, loan portfolio health, risk indicators, and business growth opportunities."
        
        # For all report types except text-summary
        if report_type != 'text-summary':
            # Create the analysis prompt
            prompt = f"""
            You are a financial analyst for Business Banking at BPI. Analyze the following financial document and extract key insights relevant for business banking.
            
            Document Content:
            {text_content[:10000]}  # Limit text length
            
            {analysis_focus}
            
            Please provide a detailed, well-structured analysis with the following sections:
            
            ## Executive Summary
            [Provide a concise 3-10 sentence summary of the key findings and their business implications]
            
            ## Key Metrics and Findings
            - Financial Metrics: [List 3-10 key financial metrics found in the document with their values]
            - Performance Indicators: [List 3-10 performance indicators with their significance]
            - Risk Assessment: [Identify key risk factors and their potential impact]
            
            ## Recommendations
            [Provide 3-10 actionable recommendations based on the analysis]
            
            ## Opportunities for Banking Services
            [Suggest 2-5 specific banking products or services that would benefit the business based on the analysis]
            
            Format your response with clear headings and bullet points for readability.
            """
            
            # Send to LLM for analysis
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in business banking report generation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            analysis_text = response.choices[0].message.content
            
            # Fix the metrics extraction to use a proper JSON structure
            metrics_prompt = """
            Based on the document content I provided, extract exactly 4 quantitative metrics that would be important for business banking purposes.
            
            Return your response ONLY as a valid JSON object with this exact structure:
            {
                "metrics": [
                    {"name": "Metric 1 Name", "value": 123, "unit": "%"},
                    {"name": "Metric 2 Name", "value": 456, "unit": "$M"},
                    {"name": "Metric 3 Name", "value": 789, "unit": "ratio"},
                    {"name": "Metric 4 Name", "value": 101, "unit": "days"}
                ]
            }
            
            Use numeric values only for the "value" field. If you cannot extract real metrics, provide reasonable estimates.
            """
            
            metrics_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant that responds only with valid JSON."},
                    {"role": "user", "content": metrics_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            try:
                # Try to extract metrics from JSON
                metrics_text = metrics_response.choices[0].message.content
                metrics_data = json.loads(metrics_text)
                metrics = metrics_data.get('metrics', [])
                if not metrics or len(metrics) < 4:
                    raise ValueError("Invalid metrics format")
            except Exception as metrics_error:
                print(f"Error parsing metrics: {metrics_error}")
                # Fallback with dummy metrics
                metrics = [
                    {"name": "Revenue Growth", "value": 8.7, "unit": "%"},
                    {"name": "Profit Margin", "value": 12.3, "unit": "%"},
                    {"name": "Debt-to-Equity", "value": 1.4, "unit": "ratio"},
                    {"name": "Cash Flow", "value": 2.8, "unit": "$M"}
                ]

        # Generate and save a visualization
        chart_path = generate_metrics_chart(metrics)
        
        # Generate PDF report
        report_path = generate_pdf_report(
            analysis_text, 
            chart_path, 
            os.path.basename(file_path),
            settings
        )
        
        # Get the relative URL for the PDF
        report_url = f"/download_report/{os.path.basename(report_path)}"
        
        # Create a summary for the chat message
        summary = "I've analyzed your financial document and generated a comprehensive report. The report includes an executive summary, key metrics, recommendations, and opportunities for banking services."
        
        return {
            "success": True,
            "summary": summary,
            "details": analysis_text,
            "reportUrl": report_url,
            "metrics": metrics
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Error analyzing document: {str(e)}"}

# function to generate a metrics chart
def generate_metrics_chart(metrics):
    # create a bar chart
    plt.figure(figsize=(10, 6))
    
    names = [m['name'] for m in metrics]
    values = [m['value'] for m in metrics]
    units = [m['unit'] for m in metrics]
    
    # create the bar chart
    bars = plt.bar(names, values, color='#b70c1a')
    
    # customize the chart
    plt.title('Key Financial Metrics')
    plt.ylabel('Value')
    plt.xticks(rotation=15, ha='right')
    
    # add value labels on top of bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"{values[i]}{units[i]}",
            ha='center'
        )
    
    plt.tight_layout()
    
    # save the chart to a file
    chart_filename = f"chart_{uuid.uuid4().hex[:8]}.png"
    chart_path = os.path.join(app.config["REPORTS_FOLDER"], chart_filename)
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

# function to generate PDF report
def generate_pdf_report(analysis_text, chart_path, original_filename, settings):
    """Generate a professional-looking PDF report with well-formatted text and proper spacing"""
    # create a unique filename for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"financial_report_{timestamp}.pdf"
    report_path = os.path.join(app.config["REPORTS_FOLDER"], report_filename)
    
    print(f"Generating PDF report at: {report_path}")
    
    try:
        # create a buffer for the PDF
        buffer = BytesIO()
        
        # create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        elements = []
        
        # enhanced styles for better text presentation
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            textColor=colors.HexColor('#730007'),
            fontSize=20,
            spaceAfter=24,
            alignment=1  # Centered
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leading=14  # Line spacing
        )
        
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=20,
            firstLineIndent=-10,
            spaceAfter=3,
            leading=14
        )
        
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            textColor=colors.HexColor('#730007'),
            fontSize=14,
            spaceBefore=12,
            spaceAfter=8
        )
        
        subsection_style = ParagraphStyle(
            'Subsection',
            parent=styles['Heading3'],
            textColor=colors.HexColor('#b70c1a'),
            fontSize=12,
            spaceBefore=8,
            spaceAfter=6
        )
        
        date_style = ParagraphStyle(
            'Date',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.gray,
            spaceAfter=4
        )
        
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=1  # Centered
        )
        
        # add title
        elements.append(Paragraph("Financial Analysis Report", title_style))
        
        # add date and file info
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y, %H:%M')}", date_style))
        elements.append(Paragraph(f"Source document: {original_filename}", date_style))
        elements.append(Spacer(1, 20))
        
        # clean up the analysis text before processing
        # this helps with markdown formatting
        def clean_markdown(text):
            text = text.replace('- ', '\n- ')
            text = text.replace('• ', '\n- ')
            text = text.replace('* ', '\n- ')
            
            text = text.replace('\n##', '\n## ')
            text = text.replace('\n###', '\n### ')
            
            # remove extra line breaks that might affect formatting
            while '\n\n\n' in text:
                text = text.replace('\n\n\n', '\n\n')
                
            return text
        
        analysis_text = clean_markdown(analysis_text)
        sections = analysis_text.split('## ')
        
        for section_idx, section in enumerate(sections):
            if not section.strip():
                continue
                
            # extract section title and content
            parts = section.split('\n', 1)
            if len(parts) == 2:
                section_title = parts[0].strip()
                section_content = parts[1].strip()
                
                # add section title (but not for the first section if it's empty)
                if section_idx > 0 or section_title:
                    elements.append(Paragraph(section_title, section_style))
                
                # process subsections (###) if any
                if '### ' in section_content:
                    subsections = section_content.split('### ')
                    for subsection in subsections:
                        if not subsection.strip():
                            continue
                            
                        subsection_parts = subsection.split('\n', 1)
                        if len(subsection_parts) == 2:
                            subsection_title = subsection_parts[0].strip()
                            subsection_content = subsection_parts[1].strip()
                            
                            # add subsection title
                            elements.append(Paragraph(subsection_title, subsection_style))
                            
                            # process content with bullet points
                            process_content_with_bullets(subsection_content, elements, bullet_style, normal_style)
                else:
                    # process content with bullet points
                    process_content_with_bullets(section_content, elements, bullet_style, normal_style)
                
                elements.append(Spacer(1, 10))
            elif section.strip():
                # if there's only content without a title
                process_content_with_bullets(section, elements, bullet_style, normal_style)
                elements.append(Spacer(1, 10))
        
        # add metrics visualization if it exists
        if os.path.exists(chart_path):
            elements.append(Paragraph("Key Metrics Visualization", section_style))
            elements.append(Spacer(1, 8))
            
            # add the image
            img = Image(chart_path, width=450, height=270)
            elements.append(img)
            elements.append(Spacer(1, 20))
        
        # Add disclaimer
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("Disclaimer: This report is generated automatically based on the provided document. The analysis should be reviewed by a financial professional before making business decisions.", disclaimer_style))
        
        # Build the PDF
        doc.build(elements)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        # Save the PDF to a file
        with open(report_path, 'wb') as f:
            f.write(pdf_content)
        
        # Verify the file was created successfully
        if not os.path.exists(report_path):
            print(f"ERROR: PDF file was not created at {report_path}")
        else:
            print(f"Successfully created PDF report: {report_path} ({os.path.getsize(report_path)} bytes)")
            
        return report_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error generating PDF report: {str(e)}")
        
        # Create a simple error PDF if the main report generation fails
        try:
            # Create a simple buffer for the error PDF
            error_buffer = BytesIO()
            error_doc = SimpleDocTemplate(error_buffer, pagesize=letter)
            error_elements = []
            
            error_elements.append(Paragraph("Error Report", styles['Title']))
            error_elements.append(Spacer(1, 20))
            error_elements.append(Paragraph(f"An error occurred while generating the detailed report: {str(e)}", styles['Normal']))
            error_elements.append(Spacer(1, 10))
            error_elements.append(Paragraph("Please try again with a different file or report type.", styles['Normal']))
            
            error_doc.build(error_elements)
            
            # Save the error PDF
            with open(report_path, 'wb') as f:
                f.write(error_buffer.getvalue())
                
            error_buffer.close()
            return report_path
            
        except:
            # If even the error PDF fails, return a path anyway so the code doesn't break
            return report_path

# Helper function to process content with bullet points        
def process_content_with_bullets(content, elements, bullet_style, normal_style):
    """Process content text and properly format bullet points and paragraphs"""
    if '\n- ' in content:
        paragraphs = content.split('\n')
        
        current_paragraph = []
        
        for paragraph in paragraphs:
            trimmed = paragraph.strip()
            if not trimmed:
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    elements.append(Paragraph(para_text, normal_style))
                    current_paragraph = []
            elif trimmed.startswith('- '):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    elements.append(Paragraph(para_text, normal_style))
                    current_paragraph = []
                
                bullet_text = trimmed[2:].strip()
                elements.append(Paragraph(f"• {bullet_text}", bullet_style))
            else:
                if not current_paragraph:
                    current_paragraph.append(trimmed)
                else:
                    current_paragraph.append(trimmed)
        
        if current_paragraph:
            para_text = ' '.join(current_paragraph)
            elements.append(Paragraph(para_text, normal_style))
    else:
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                elements.append(Paragraph(para.strip(), normal_style))

@app.route("/generate_report", methods=["POST"])
def handle_report_generation():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"})
            
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})
        
        # Parse settings
        settings = {}
        if 'settings' in request.form:
            try:
                settings = json.loads(request.form['settings'])
            except json.JSONDecodeError:
                # Fallback to default settings if JSON parsing fails
                settings = {"reportType": "comprehensive", 
                           "includeExecutiveSummary": True,
                           "includeKeyMetrics": True, 
                           "includeRecommendations": True}
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        # Analyze the document
        result = analyze_financial_document(file_path, settings)
        
        if "error" in result:
            return jsonify({"error": result["error"]})
        
        # Add message to chat history
        add_message("bot", f"## Financial Report Generated\n\n{result['details']}")
        
        return jsonify(result)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

# Route to download generated report
@app.route("/download_report/<filename>", methods=["GET"])
def download_report(filename):
    try:
        return send_file(
            os.path.join(app.config["REPORTS_FOLDER"], filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"error": f"Error downloading report: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True) 
