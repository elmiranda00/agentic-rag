import os
import re
import io
import tempfile
import base64
from typing import List
from collections import OrderedDict
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

# Streamlit imports
import streamlit as st
from streamlit_chat import message

# LangChain imports
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.base import Tool
from langchain.tools import StructuredTool
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains.llm import LLMChain

from langchain.agents import initialize_agent, AgentType

# Helper function to sanitize uploaded file names
def safe_filename(name):
    return re.sub(r'[^\w\-_.]', '_', name)


class ChatPDFAssistant:
    """Handles multiple data types, web search, and data analysis with agent-based workflow"""
    
    def __init__(self):
        
        llm_api_key = os.getenv("GROQ_API_KEY")
        if not llm_api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment.")
        self.model = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                                 groq_api_key=llm_api_key,
                                 temperature= 0)
        self.embeddings =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        #os.environ["HTTP_PROXY"] = "http://your-proxy.com"
        #os.environ["HTTPS_PROXY"] = "http://your-proxy.com"

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self.prompt = self._create_prompt_template()
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.datasets = {}
        self.tools = self._initialize_tools()
        self.agent_executor = self._create_agent_executor()
        #self.structured_tool = self._create_structured_tool()

        serp_api_key = os.getenv("SERP_API_KEY")
        if not serp_api_key:
            raise ValueError("SERP_API_KEY is not set in the environment.")
        self.serp_api = SerpAPIWrapper(serpapi_api_key=serp_api_key)
    
    def _initialize_tools(self):
        return [
            # 1. RAG tool
            Tool(
                name="RAG_Query",
                func=self._rag_query,
                description="Always try this tool **first** for any question related to the uploaded PDF document."
                    "Use it when the user asks a question that may be answered from the PDF. ",
                return_direct=True 
            ),

            # 2. Web Search tool -- disabled
            #Tool(
            #    name="Web_Search",
            #    func=self._web_search,
            #    description="Use when PDF content does not contain the answer or is insufficient. "
            #        "Best for real-time or external web information that is not present in the PDF."
            #),

            # 3. Data Analysis tool
            self._create_structured_tool()
        ]
    
    def _create_agent_executor(self):
        return initialize_agent(
            self.tools,
            self.model,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            system_message=self._create_prompt_template()
        )
    
    def _create_structured_tool(self):

        class DataAnalysisInput(BaseModel):
            dataset_name: str
            query: str

        return StructuredTool.from_function(
            name="Data_Analysis",
            func= self._analyze_data_structured,
            description= 
                """Use this tool for any kind of data analysis, including:
                - statistical summaries
                - filtering or grouping
                - visualizations like line plots, bar charts, histograms

                Input should include:
                - `dataset_name`: the uploaded dataset (e.g., "sales.csv")
                - `query`: a plain English request like "plot revenue over time" or "show average age by region"

                Expected output:
                - `result`: plain text summary or analysis
                - `plot`: a base64-encoded PNG image if a chart is created, or `null` otherwise
                - If displaying a table, convert it to a readable string using `.to_string(index=False)` instead of `.to_csv()` or raw string formatting.
                - If a plot is generated, do not call `plt.show()` — just create the plot.
                """,
            args_schema= DataAnalysisInput
        )

    @staticmethod
    def _create_prompt_template():
        return ChatPromptTemplate.from_template(
            """
            <s> [INST] 
            You are an intelligent AI assistant with access to PDF documents (via RAG), datasets, and live web search.

            **Guidelines:**
            1. **Always attempt to answer using the the PDFs or datasets first.**
                - Use all partial or related information available in the documents.
                - Do not fabricate details if precise answers are missing.
                - Example: “The report does not provide a full breakdown, but includes the following details...”
            2. **Use web search to **supplement** (not replace) the answer only if the PDF lacks some information or is not sufficient**
                - Clearly separate what came from the document vs the web.
                - Do not ignore document context unless it's completely irrelevant.
            3. **For data-related questions (e.g., 'total sales by year')**, check if a dataset is available.
                - If yes, analyze it.
                - If not, ask the user to upload one.
            4. **Use markdown formatting.**
            5. **Be concise, accurate, and never guess facts. Combine sources when helpful.**
            6. "If a tool provides source references (e.g., [1], [2], [3]), include those links at the end of your Final Answer."
            [/INST] </s>
            [INST]
            Question: {question}
            Context: {context}
            Tools available: RAG (PDF), Data Analysis, Web Search
            Answer (Only answer using the text from the context above. Do not make assumptions or fabricate information):
            [/INST]
            """
        ) 

    
    def _rag_query(self, query: str) -> str:
        if not self.vector_store:
            return "Please upload data sources first."
        
        # Build context from retrieved docs
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.5}     
        )

        relevant_docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in relevant_docs])
        context = "\n\n".join(
            f"{d.page_content.strip()} [Source: {d.metadata.get('source', 'Unknown')}, p.{d.metadata.get('page', 'N/A')}]"
            for d in relevant_docs
        )

        # Introduce a fallback hint in case sufficient info is not found in the docs
        if not context.strip() or len(relevant_docs) < 2:
            fallback_hint = (
                "Note: the full structured answer may not be available. "
                "Summarize whatever partial mentions or related data you find."
            )
            context = fallback_hint + "\n\n" + context

        # Get response from the model
        self.chain = LLMChain(llm=self.model, prompt=self.prompt)
        answer = self.chain.run(question=query, context=context)

        #self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
        #         | self._create_prompt_template()
        #         | self.model
        #         | StrOutputParser())
        #answer = self.chain.invoke(query)
        
        # Build source annotation
        source_map = OrderedDict()
        
        #print("Retrieved:", [doc.page_content[:100] for doc in relevant_docs]) #DEBUGGING

        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            key = f"{source} (p.{page})"
            if key not in source_map:
                source_map[key] = len(source_map) + 1
        # Insert citations in text
        for key, num in source_map.items():
            marker = f"[{num}]"
            # TBD - improve by associating marker with specific facts or sections (NLP matching etc)
            if marker not in answer:
                answer += f" {marker}"

        source_annotation = "\n\nSources:\n" + "\n".join([f"[{num}] {src}" for src, num in source_map.items()])
        
        return "Final Answer:\n" + answer.strip() + source_annotation
    
    
    #def _prepare_retriever(self):
    #    self.retriever = self.vector_store.as_retriever(
    #        search_type="similarity_score_threshold",
    #        search_kwargs={"k": 3, "score_threshold": 0.5},
    #    )
    #    self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
    #                  | self.prompt
    #                  | self.model
    #                  | StrOutputParser())

    def _web_search(self, query: str) -> str:
        
        try:
            results = self.serp_api.results(query)  # returns a dict
            organic = results.get("organic_results", [])
            if not organic:
                return "No relevant web results found."
            # Extract top 3 results
            sources = [
                f"[{i+1}] {item.get('title', 'No Title')} - {item.get('link', 'No Link')}"
                for i, item in enumerate(organic[:3])
            ]
            # Get the top snippet (or first one with a snippet)
            snippet = next((item.get("snippet") for item in organic if "snippet" in item), "No summary available.")
            # Return snippet with citations
            return f"{snippet} " + " ".join(f"[{i+1}]" for i in range(len(sources))) + "\n\nSources:\n" + "\n".join(sources)

        except Exception as e:
            return f"Web search failed: {str(e)}"
        

    # OLD
    def _analyze_data(self, query: str) -> str:
        if not self.datasets:
            return "No datasets available. Please upload data first."
        
        # Code generation and safe execution
        code = self._generate_analysis_code(query)
        return self._safe_execute_code(code)
    
    # OLD
    def _generate_analysis_code(self, query: str) -> str:
        prompt = f"""Generate Python code to analyze datasets: {list(self.datasets.keys())}
        Query: {query}
        Return ONLY valid Python code with NO text explanation."""
        return self.model.invoke(prompt).content
    
    
    def _extract_code(self, text: str) -> str:
        """Extract code from a markdown fenced block or return raw if not fenced"""
        match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()
    
    def _analyze_data_structured(self, dataset_name: str, query: str) -> str:
        df = self.datasets.get(dataset_name)
        columns = ", ".join(df.columns) if df is not None else "Unknown"
        if df is None:
            return f"Dataset '{dataset_name}' not found."
        
        prompt = f"""
            You are given a pandas DataFrame named `datasets['{dataset_name}']`.
            It has the following columns: {columns}
            The user wants to: {query}

            Write Python code that uses this dataset to perform the task.
            Set your final output to a variable called `result`.

            If displaying a table, convert it to a readable string using `.to_string(index=False)` instead of `.to_csv()` or raw string formatting.
            If a plot is generated, do not call `plt.show()` — just create the plot.

            Return ONLY valid Python code (no explanations or markdown)."""
        
        code = self._extract_code(self.model.invoke(prompt).content)
        return self._safe_execute_code(code)

    def _safe_execute_code(self, code: str) -> dict:
        try:
            # Restricted execution environment
            import matplotlib.pyplot as plt
            import base64
            import io

            local_vars = {"datasets": self.datasets}
            full_code = (
                "import pandas as pd\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "result = None\n"
                + code
            )

            # Capture plot
            plt.ioff() 
            #fig = plt.figure()
            
            exec(full_code, {}, local_vars)
            result = str(local_vars.get('result', '')).strip() or "Analysis complete."

            base64_img = None
            if plt.get_fignums():
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                base64_img = base64.b64encode(buf.read()).decode("utf-8")
                plt.close("all")

            return {"result": result, "plot": base64_img}

        except Exception as e:
            return {"result": f"Error in analysis: {str(e)}", "plot": None}
    
    def ingest_file(self, file_path: str, file_type: str):
        if file_type == "pdf":
            self.ingest_pdf(file_path)
        elif file_type in ["csv", "xlsx"]:
            self.ingest_data(file_path, file_type)
    
    def ingest_pdf(self, pdf_file_path: str):
        all_chunks = []
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        for i, doc in enumerate(docs):
            doc.metadata["page"] = i + 1
            doc.metadata["source"] = os.path.basename(pdf_file_path)

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        all_chunks.extend(chunks)
        
        self.vector_store = FAISS.from_documents(documents=all_chunks, 
                                                  embedding=self.embeddings)
    
    def ingest_data(self, file_path: str, file_type: str):
        dataset_name = os.path.basename(file_path)
        if file_type == "csv":
            self.datasets[dataset_name] = pd.read_csv(file_path)
        elif file_type == "xlsx":
            self.datasets[dataset_name] = pd.read_excel(file_path)


    def ask(self, query: str) -> dict:
        #if not self.vector_store and not self.datasets:
        #    return "Please, add data sources (PDF documents, Excel/CSV files) first."
        #return self.agent_executor.run(query)
    
        if not self.vector_store and not self.datasets:
            return {"result": "Please, add data sources (PDF documents, Excel/CSV files) first.", "plot": None}
    
        try:
            response = self.agent_executor.invoke({"input": query})
            if isinstance(response, dict) and "output" in response:
                response = response["output"]
        except Exception as e:
            return {"result": f"Agent error: {str(e)}", "plot": None}
    
        # If the analysis step was triggered, _safe_execute_code returned a dict already.
        if isinstance(response, dict) and "result" in response:
            return response
        elif isinstance(response, str):
            # crude check for inline encoded image base64 string
            if "data:image/png;base64," in response:
                return {"result": "", "plot": response}
            return {"result": response.strip(), "plot": None}
        else:
            return {"result": str(response), "plot": None}
        
            #return response
        #else:
            #return {"result": str(response), "plot": None}

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

#==================== Streamlit Code =====================

def init_page() -> None:
    st.set_page_config(
        page_title="Insight Agent"
    )
    st.sidebar.title("Data Source Manager")
    st.sidebar.markdown("Upload or clear documents here.")

def init_session():
    if "assistant" not in st.session_state:
        st.session_state.assistant = ChatPDFAssistant()
    if "messages" not in st.session_state:
        st.session_state.messages = []
  

def handle_file_upload():
    files = st.session_state.get("file_uploader", [])
    if not files:
        return

    if "assistant" in st.session_state:
        st.session_state["assistant"].clear()
    st.session_state["messages"] = []

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    for file in files:
        file_ext = os.path.splitext(file.name)[1].lower()
        file_type = "pdf" if file_ext == ".pdf" else "csv" if file_ext == ".csv" else "xlsx"

        safe_name = safe_filename(file.name)
        save_path = os.path.join(upload_dir, safe_name)

        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

        with st.spinner(f"Ingesting {file.name}..."):
            st.session_state["assistant"].ingest_file(save_path, file_type)
        
        #with tempfile.NamedTemporaryFile(delete=False) as tmp:
        #    tmp.write(file.getbuffer())
        #    tmp_path = tmp.name
        
        #with st.spinner(f"Ingesting {file.name}..."):
        #    st.session_state["assistant"].ingest_file(tmp_path, file_type)
        
        #os.remove(tmp_path)

def setup_sidebar():
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload new document", 
        type=["pdf", "csv", "xlsx"], 
        key="file_uploader", 
        accept_multiple_files=True,
        label_visibility="collapsed",
        on_change=handle_file_upload
    )

    if st.sidebar.button("Clear Documents"):
        if "assistant" in st.session_state:
            st.session_state["assistant"].clear()
        st.session_state["messages"] = []


def main() -> None:
    
    load_dotenv(find_dotenv())

    init_page()
    setup_sidebar()
    init_session()

    st.header("Insight Agent")
    st.subheader("Chat with your reports and datasets")

    #Handle user query
    user_text = st.chat_input("Input your question!")
    if user_text:
        user_text = user_text.strip()
        if user_text:
            st.session_state["messages"].append((user_text, True))
            #st.session_state["user_input"] = ""
            with st.spinner("Insight Agent is thinking..."):
                try:
                    agent_reply = st.session_state["assistant"].ask(user_text)
                except Exception as e:
                    agent_reply = {"result": f"Error: {str(e)}", "plot": None}
            st.session_state["messages"].append((agent_reply, False))
    
    # Display chat history
    for message_text, is_user in st.session_state["messages"]:
        with st.chat_message("user" if is_user else "assistant"):
            if is_user:
                st.markdown(message_text)
            else:
                st.markdown(message_text["result"])
                if message_text["plot"]:
                    #st.image(f"data:image/png;base64,{message_text['plot']}", use_column_width=True)
                    st.markdown(f"![Plot](data:image/png;base64,{message_text['plot']})", unsafe_allow_html=True)

# streamlit run app.py
if __name__ == "__main__":
    main()