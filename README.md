# Bria - BPI Business Banking Assistant

Bria is an AI-powered assistant designed specifically for BPI Business Banking. This application provides intelligent banking tools for business clients, including automated reporting, risk assessment, query handling, document analysis, and industry classification.

## Features

### 1. Automated Report Generation
Generate comprehensive financial reports from uploaded documents with various analysis types:
- **Comprehensive Analysis**: Full overview of financial performance, risk indicators, and growth opportunities
- **Loan Portfolio Analysis**: Focus on outstanding balances, risk levels, and portfolio diversity
- **Financial Metrics Analysis**: Key metrics including revenue, expenses, and profitability ratios
- **Campaign Performance Analysis**: Evaluate marketing campaign metrics like conversion rates and ROI
- **Text Summarization**: Simple document summarization for non-financial documents

### 2. Risk Assessment
Analyze client financial profiles to evaluate creditworthiness, predict default risks, and identify potential high-value leads.

### 3. Intelligent Query Handling
Use natural language processing to handle client inquiries regarding loan eligibility, credit applications, and campaign details.

### 4. Document Analysis
Upload and analyze various document types:
- PDF files
- Excel spreadsheets
- CSV data
- Parquet files
- Images (with text extraction)

### 5. Automated Industry Classification
Analyze transaction data to automatically classify businesses into appropriate industry categories, with:
- Primary industry identification
- Sub-industry classification
- Confidence level assessment
- Key transaction indicators
- Alternative classifications
- Recommended financial products

## Technical Details

### Architecture
- **Backend**: Flask-based Python application
- **Frontend**: HTML, CSS, and JavaScript with responsive design
- **AI Integration**: OpenAI's GPT models for intelligence
- **Vector Database**: ChromaDB for document storage and retrieval
- **External API Integration**: Tavily for web search capabilities

### AI Capabilities
- Natural language understanding and generation
- Document analysis and summarization
- Intelligent search across uploaded documents
- Web search for additional context
- Financial data extraction and interpretation
- Transaction pattern analysis for industry classification

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Tavily API key (optional, for web search)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bpi-business-banking-bria.git
cd bpi-business-banking-bria
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here  # Get from https://tavily.com/
   ```

4. Install Tesseract OCR for image processing:
   ```
   # macOS
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # Windows
   # Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

5. Create a `.env` file based on the `.env.example` file:
   ```
   cp .env.example .env
   ```

## Running the Application

1. Run the test script to check your environment:
   ```
   python test_app.py
   ```

2. Start the Flask server:
   ```
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

4. Upload documents and start chatting with the advanced assistant!

## Troubleshooting

If you encounter any issues:

1. **LangChain Imports**: Make sure you're using the correct import paths. The application uses:
   - `langchain_openai` for OpenAI models and embeddings
   - `langchain_chroma` for vector database
   - `langchain_community` for document loaders and tools

2. **API Key Issues**: Ensure your `.env` file has the correct API keys:
   - OPENAI_API_KEY is required for all functionality
   - TAVILY_API_KEY is required for web search features

3. **Missing Dependencies**: Run `pip install -r requirements.txt` to install all dependencies

4. **File Processing Issues**: Ensure Tesseract OCR is installed if you plan to process images

## Customization

- Modify system prompts and agent instructions in `app.py`
- Adjust the vector store settings to optimize for different document types
- Customize the UI by editing the CSS in `static/styles.css`
- Add support for additional file formats by creating new processing functions 
