# Agentic RAG

This Streamlit app provides an agent-based Retrieval-Augmented Generation (RAG) workflow for querying PDFs and analyzing datasets using LLMs.
You can use it to chat with your with uploaded PDFs and datasets (CSVs) and perform simple data analysis on Excel files (XLSX). The output is enhanced with a web search if needed.

---

## Setup

### 1. Environment Variables
Create a `.env` file with:
GROQ_API_KEY=your_groq_key
SERP_API_KEY=your_serp_key

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the App
streamlit run app.py
