import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Check if the environment is properly set up."""
    print("Checking environment...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Error: .env file not found. Please create one based on .env.example")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Check if required API keys are set
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        return False
    
    if not tavily_key:
        print("Warning: TAVILY_API_KEY not found in .env file. Web search will not work.")
    
    # Check if uploads directory exists
    if not os.path.exists('uploads'):
        print("Creating uploads directory...")
        os.makedirs('uploads', exist_ok=True)
    
    # Check if chromadb_store directory exists
    if not os.path.exists('chromadb_store'):
        print("Creating chromadb_store directory...")
        os.makedirs('chromadb_store', exist_ok=True)
    
    print("Environment check complete. No critical issues found.")
    return True

def check_imports():
    """Try importing required packages to validate installation."""
    print("Checking required packages...")
    
    try:
        import flask
        import openai
        import langchain
        import langchain_openai
        import langchain_chroma
        import pandas
        import chromadb
        import PyPDF2
        import PIL
        import pytesseract
        print("All required packages are installed.")
        return True
    except ImportError as e:
        print(f"Error: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    env_ok = check_environment()
    imports_ok = check_imports()
    
    if env_ok and imports_ok:
        print("System is ready to run the application.")
        print("Start the application with: python app.py")
    else:
        print("Please fix the issues above before running the application.") 