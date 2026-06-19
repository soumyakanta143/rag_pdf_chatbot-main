# 📚 Local RAG PDF Chatbot

An interactive PDF chatbot built with **Streamlit**, **LangChain**, and a **Local MCP Cost-Orchestrator** that allows users to **chat with multiple PDFs** entirely locally with **no manual API keys required**.

---

## 🚀 Optimized Features
- 📄 **Upload & Process PDFs**: Process multiple PDFs in parallel.
- 🏎️ **Persistent MCP Server (Low Latency)**: Reuses a single running Node process in the background. Handshakes are executed only once, saving **1+ seconds** on every message.
- 💾 **Local Document Cache**: Automatically saves processed text chunks to `indexed_cache.json`. When you restart the app, it loads your notes instantly so you don't have to re-upload the PDFs.
- 🔎 **Hybrid Search (Semantic & TF-IDF)**:
  - Automatically performs **Semantic Vector Similarity Search** using a local `sentence-transformers` model (`all-MiniLM-L6-v2`) if installed.
  - Gracefully falls back to a pure-Python **Lexical TF-IDF** index if the package is missing, displaying your active search type in the sidebar.
- ⬇️ **Export History**: Download full chat transcripts as CSV.

---

## 🛠️ Tech Stack
- **Frontend & App Logic:** Streamlit, Pandas
- **Orchestration:** LangChain (RecursiveCharacterTextSplitter)
- **Local Search Engine:** Semantic (SentenceTransformers) with TF-IDF fallback
- **Subprocess Protocol Client:** Standard `subprocess` JSON-RPC stdio
- **PDF Extraction:** PyPDF2

---

## ⚙️ Setup & Running Locally

### 1. Create and Activate a Virtual Environment
```bash
python -m venv myenv
# On Windows (PowerShell):
.\myenv\Scripts\Activate.ps1
# On Linux/macOS:
source myenv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Local MCP Server Setup
The application reads your local MCP settings dynamically from `~/.gemini/antigravity-ide/mcp_config.json`. Ensure the `antigravity-cost-orchestrator` is configured:
```json
{
  "mcpServers": {
    "antigravity-cost-orchestrator": {
      "command": "node",
      "args": [
        "c:/Users/sahus/OneDrive/Documents/mcp/index.js",
        "--mcp"
      ],
      "env": {
        "OPENROUTER_API_KEY": "your-free-tier-openrouter-key",
        "GROQ_API_KEY": "your-groq-key"
      }
    }
  }
}
```

### 4. Start the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. Upload your files, click **Submit & Process**, and start chatting!
