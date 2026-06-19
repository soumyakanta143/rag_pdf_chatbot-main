import os
import base64
import asyncio
from datetime import datetime

import streamlit as st
import pandas as pd


# LangChain Text Splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Event loop check (asyncio)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import re
import math
import json
import subprocess
from collections import Counter

# ---------------------- Local Search Engine & Retrievers ---------------------- #
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class SimpleTFIDFRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vocab = set()
        self.doc_counts = []
        self.idf = {}
        
        # Tokenize and count terms for each chunk
        for chunk in chunks:
            tokens = self._tokenize(chunk)
            self.doc_counts.append(Counter(tokens))
            self.vocab.update(tokens)
        
        # Compute IDF for all terms in the vocabulary
        num_docs = len(chunks)
        for term in self.vocab:
            doc_freq = sum(1 for counts in self.doc_counts if term in counts)
            self.idf[term] = math.log((1 + num_docs) / (1 + doc_freq)) + 1

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def retrieve(self, query, k=5):
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx, counts in enumerate(self.doc_counts):
            score = 0
            for term in query_tokens:
                if term in counts:
                    # Term frequency normalization
                    tf = counts[term] / max(1, sum(counts.values()))
                    score += tf * self.idf.get(term, 0)
            scores.append((score, self.chunks[idx]))
        
        # Sort chunks by relevance score in descending order
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Map output to standard LangChain Document schema format
        try:
            from langchain_core.documents import Document
        except ImportError:
            try:
                from langchain.schema import Document
            except ImportError:
                from langchain.docstore.document import Document
        return [Document(page_content=chunk) for score, chunk in scores[:k]]

class SemanticRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        # Load a small, lightweight semantic model (runs completely free on CPU)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(chunks, convert_to_numpy=True)

    def retrieve(self, query, k=5):
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        # Cosine similarity calculation
        norm_embeddings = np.linalg.norm(self.embeddings, axis=1)
        norm_query = np.linalg.norm(query_embedding)
        similarities = np.dot(self.embeddings, query_embedding) / (norm_embeddings * norm_query + 1e-9)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        try:
            from langchain_core.documents import Document
        except ImportError:
            try:
                from langchain.schema import Document
            except ImportError:
                from langchain.docstore.document import Document
        return [Document(page_content=self.chunks[idx]) for idx in top_indices]

class SimpleRetrieverWrapper:
    def __init__(self, chunks):
        self.chunks = chunks
        self.is_semantic = False
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.retriever = SemanticRetriever(chunks)
                self.is_semantic = True
            except Exception:
                self.retriever = SimpleTFIDFRetriever(chunks)
        else:
            self.retriever = SimpleTFIDFRetriever(chunks)

    def similarity_search(self, query, k=5):
        return self.retriever.retrieve(query, k=k)

    def max_marginal_relevance_search(self, query, k=5, fetch_k=15):
        return self.retriever.retrieve(query, k=k)


import requests

import time

# ---------------------- Cloud-Friendly LLM Client ---------------------- #
def get_llm_answer(formatted_prompt, system_prompt):
    # Retrieve API key from environment (or Streamlit Secrets in production)
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    if not OPENROUTER_API_KEY:
        try:
            OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
        except Exception:
            pass
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # We use a completely free model on OpenRouter
    data = {
        "model": "cohere/north-mini-code:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
    }
    
    max_retries = 3
    base_delay = 10 # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt)) # Exponential backoff: 2s, 4s...
            else:
                # If all retries fail, return a graceful message instead of a raw Exception
                return "⚠️ **The AI server is currently experiencing high traffic and couldn't process your request. Please wait a minute and try asking again.**"

import io
import csv

# ---------------------- Helpers ---------------------- #
def extract_text_from_files(files):
    text = ""
    for f in files:
        file_extension = f.name.split('.')[-1].lower()
        if file_extension == 'pdf':
            try:
                import pdfplumber
                with pdfplumber.open(f) as pdf:
                    for page in pdf.pages:
                        # pdfplumber preserves layout and reading order just like PyMuPDF but is pure Python
                        page_text = page.extract_text() or ""
                        text += page_text + "\n"
            except ImportError:
                st.error("pdfplumber is not installed. Skipping PDF document.")
            except Exception as e:
                st.warning(f"Failed to parse PDF {f.name}: {e}")
        elif file_extension == 'docx':
            try:
                import docx
                doc = docx.Document(f)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except ImportError:
                st.error("python-docx is not installed. Skipping Word document.")
        elif file_extension == 'csv':
            try:
                import pandas as pd
                df = pd.read_csv(f)
                text += df.to_string() + "\n"
            except Exception:
                content = f.getvalue().decode("utf-8")
                reader = csv.reader(io.StringIO(content))
                for row in reader:
                    text += " ".join(row) + "\n"
        elif file_extension == 'txt':
            text += f.getvalue().decode("utf-8") + "\n"
    return text


def split_text(text):
    # Smart chunking with semantic separators to avoid splitting sentences/paragraphs
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    st.sidebar.info(f"DEBUG: {len(chunks)} chunks created")
    return chunks

CACHE_FILE = "indexed_cache.json"

def save_cache(chunks):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"chunks": chunks}, f)
    except Exception as e:
        st.sidebar.warning(f"Failed to save document cache: {e}")

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
            chunks = data.get("chunks", [])
            if chunks:
                return SimpleRetrieverWrapper(chunks)
        except Exception:
            pass
    return None

def build_vectorstore(files):
    raw_text = extract_text_from_files(files)
    if not raw_text.strip():
        raise ValueError("No extractable text found in the uploaded document(s).")

    chunks = split_text(raw_text)

    # Save to local file cache
    save_cache(chunks)

    vs = SimpleRetrieverWrapper(chunks)
    st.session_state["vectorstore"] = vs
    
    # Save the filenames for the UI feedback
    st.session_state["loaded_files"] = [f.name for f in files]
    return vs

def condense_question(history_str, question):
    if history_str:
        prompt_text = (
            f"Given the following conversation history and a question, rewrite the question into 3 different search query variations "
            f"to maximize the chances of finding relevant documents. Return the 3 variations separated by newlines, with no other text or numbering.\n\n"
            f"Chat History:\n{history_str}\nQuestion: {question}\n\nStandalone Queries:"
        )
    else:
        prompt_text = (
            f"Given the following question, rewrite it into 3 different search query variations using different keywords/synonyms "
            f"to maximize the chances of finding relevant documents. Return the 3 variations separated by newlines, with no other text or numbering.\n\n"
            f"Question: {question}\n\nSearch Queries:"
        )
        
    try:
        standalone = get_llm_answer(prompt_text, "You are a helpful assistant that reformulates search queries.")
        if standalone and not standalone.startswith("Error"):
            return standalone.strip()
    except Exception:
        pass
    return question

def answer_question(question):
    if "vectorstore" not in st.session_state:
        st.warning("Please upload documents and click 'Process & Build Index' first.")
        return ""

    # Condense question based on chat history to support follow-up questions
    history_str = ""
    if "messages" in st.session_state and st.session_state["messages"]:
        for m in st.session_state["messages"][-4:]:
            role = "User" if m["role"] == "user" else "Assistant"
            history_str += f"{role}: {m['content']}\n"
            
    search_query = condense_question(history_str, question)
    st.session_state["last_search_query"] = search_query
    
    vs = st.session_state["vectorstore"]

    search_queries = [q.strip() for q in search_query.split('\n') if q.strip()]
    if not search_queries:
        search_queries = [question]

    all_docs = []
    seen_content = set()
    
    # Run Multi-Query Search
    for sq in search_queries:
        try:
            docs = vs.max_marginal_relevance_search(sq, k=3, fetch_k=10)
        except Exception:
            docs = vs.similarity_search(sq, k=3)
            
        for d in docs:
            if d.page_content not in seen_content:
                seen_content.add(d.page_content)
                all_docs.append(d)

    # Take the top 5 unique documents from all queries
    docs = all_docs[:5]

    st.session_state["last_retrieved_docs"] = docs

    if not docs:
        general_prompt = f"User Question: {question}"
        general_system = "You are a helpful AI assistant. Answer the user's question clearly and concisely based on your general knowledge."
        general_ans = get_llm_answer(general_prompt, general_system)
        
        if general_ans and not general_ans.startswith("Error"):
            return "⚠️ **Note: I could not find this information in your uploaded documents. The following answer is from my general AI knowledge.**\n\n" + general_ans
        return "answer is not available in the context"

    context = "\n\n".join([d.page_content for d in docs])
    
    # Pass the original question to the LLM so it has the natural question format
    formatted_prompt = f"Context Information:\n{context}\n\nUser Question: {question}\n\nPlease provide a clear answer based solely on the Context Information above."
    system_prompt = (
        "You are a helpful AI assistant. You must answer the user's question based strictly on the provided context.\n"
        "If the answer cannot be found in the context, explicitly state: 'answer is not available in the context'.\n"
        "Do not use any outside knowledge. Be concise and direct."
    )
    
    ans = get_llm_answer(formatted_prompt, system_prompt)
    
    if "answer is not available in the context" in ans.lower():
        general_prompt = f"User Question: {question}"
        general_system = "You are a helpful AI assistant. Answer the user's question clearly and concisely based on your general knowledge."
        general_ans = get_llm_answer(general_prompt, general_system)
        
        if general_ans and not general_ans.startswith("Error"):
            ans = "⚠️ **Note: I could not find this information in your uploaded documents. The following answer is from my general AI knowledge.**\n\n" + general_ans
            
    return ans


# ---------------------- UI ---------------------- #
def main():
    st.set_page_config(page_title="Nexus PDF AI", page_icon="🧠", layout="centered")
    
    # Inject Custom CSS for Premium Look
    st.markdown("""
    <style>
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Button Animations */
        .stButton button {
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
        }
        
        /* Input & Uploader Styling */
        .stFileUploader > div > div {
            background-color: #1E293B;
            border-radius: 12px;
            border: 1px dashed #475569;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Try to load cached vectorstore on startup
    if "vectorstore" not in st.session_state:
        vs = load_cache()
        if vs:
            st.session_state["vectorstore"] = vs

    # Premium Hero Section
    st.markdown("<h1 style='text-align: center; color: #F8FAFC;'>🧠 Nexus PDF AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 1.1rem; margin-bottom: 2rem;'>Upload your documents and let AI instantly find the answers for you.</p>", unsafe_allow_html=True)

    # Show active retriever status in sidebar
    st.sidebar.markdown("### ⚙️ System Status")
    if "vectorstore" in st.session_state:
        vs = st.session_state["vectorstore"]
        ret_type = "Semantic Search" if vs.is_semantic else "Lexical Search"
        st.sidebar.success(f"✅ Index Ready ({ret_type})")
        
        # Show which files are actively loaded
        if "loaded_files" in st.session_state:
            st.sidebar.caption("Currently loaded documents:")
            for filename in st.session_state["loaded_files"]:
                st.sidebar.markdown(f"- `{filename}`")
        else:
            st.sidebar.caption("Cached documents are currently loaded.")
        
        # Show retrieved docs in sidebar for troubleshooting
        if "last_retrieved_docs" in st.session_state:
            with st.sidebar.expander("🔍 Debug: View Retrieved Context", expanded=False):
                st.caption(f"**Query:** *\"{st.session_state.get('last_search_query', '')}\"*")
                for idx, doc in enumerate(st.session_state["last_retrieved_docs"]):
                    st.markdown(f"**Chunk {idx+1}:**")
                    st.text(doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""))
    else:
        st.sidebar.info("⏳ Waiting for documents...")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 1. Upload Documents")

    uploaded = st.sidebar.file_uploader(
        "Upload your files (PDF, Word, CSV, TXT) and Click on Process",
        type=["pdf", "docx", "csv", "txt"], accept_multiple_files=True,
    )

    if st.sidebar.button("⚙️ Process & Build Index", use_container_width=True):
        if not uploaded:
            st.sidebar.error("Please upload at least one document.")
        else:
            with st.spinner("Analyzing documents..."):
                try:
                    build_vectorstore(uploaded)
                    st.sidebar.success("Index ready! You can now chat.")
                    st.rerun()  # Rerun Streamlit to update the sidebar status
                except Exception as e:
                    st.sidebar.error(f"Indexing failed: {e}")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for m in st.session_state["messages"]:
        avatar = "🧑‍💻" if m["role"] == "user" else "🧠"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    # User input
    is_ready = "vectorstore" in st.session_state
    
    if not is_ready:
        st.info("👋 Welcome! Please upload and process your documents in the sidebar to start chatting.")
        
    user_q = st.chat_input("Ask a question about your documents...", disabled=not is_ready)
    
    if user_q and is_ready:
        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_q)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Analyzing context..."):
                ans = answer_question(user_q)
                st.markdown(ans)
        st.session_state["messages"].append({"role": "assistant", "content": ans})

    # Download chat history
    if st.session_state.get("messages"):
        rows = []
        for i in range(0, len(st.session_state["messages"]), 2):
            q = st.session_state["messages"][i]["content"] if i < len(st.session_state["messages"]) else ""
            a = st.session_state["messages"][i+1]["content"] if i+1 < len(st.session_state["messages"]) else ""
            rows.append((q, a, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        df = pd.DataFrame(rows, columns=["Question", "Answer", "Timestamp"])
        csv = df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        st.sidebar.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv">'
            f'<button>Download conversation history as CSV</button></a>',
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()

