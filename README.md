# NewsFindr – LLM Powered News Retrieval

AI-powered system that retrieves relevant and credible news articles using LLM-based query expansion, web search, credibility filtering, and automated summarization.

---

## Problem

Searching for reliable news can be challenging because:

- Search engines often return **duplicate or low-credibility sources**
- Important information may be **spread across multiple articles**
- Users must manually read several articles to understand the **overall story**

An intelligent system is needed to **retrieve credible news and summarize key insights automatically**.

---

## Approach

The system implements an **LLM-powered information retrieval pipeline**:

- **Query Expansion:** Uses an LLM to generate multiple related search queries for better coverage  
- **Web Search:** Retrieves news articles using DuckDuckGo search  
- **Credibility Filtering:** LLM filters out low-credibility or irrelevant articles  
- **Summarization:** LLM generates a concise summary of the final news articles  

### Pipeline

```
User Query
    ↓
LLM Query Expansion
    ↓
Web Search (DuckDuckGo)
    ↓
Credibility Filtering
    ↓
News Summarization
    ↓
Final News Insights
```

---

## Results

- Improved **search coverage** using LLM query expansion  
- Reduced noise through **LLM-based credibility filtering**  
- Generated **concise summaries** from multiple news sources  
- Demonstrates a practical **LLM-powered retrieval pipeline**

---

## Tech Stack

- Python  
- Groq API (LLaMA 3 70B)  
- DuckDuckGo Search API  
- LangChain Groq Integration  

---

## Project Structure

```
newsfindr/
│
├── newsfindr.py
├── requirements.txt
├── README.md
└── .env
```

---

## Setup

### Clone the repository

```bash
git clone https://github.com/your-username/newsfindr.git
cd newsfindr
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set your API key

Linux / Mac:

```bash
export GROQ_API_KEY="your_api_key_here"
```

Windows:

```bash
set GROQ_API_KEY=your_api_key_here
```

### Run the project

```bash
python newsfindr.py
```

---

## Example

### Input Query

```
latest developments in artificial intelligence
```

### Output

- List of credible news articles
- AI-generated summary of the retrieved articles
- Structured response containing:

```
{
  "email": "...",
  "query": "...",
  "summary": "...",
  "articles": [...]
}
```

---

## Key Insight

Combining LLM query expansion, web search, filtering, and summarization creates a powerful pipeline for retrieving reliable and concise news insights automatically.

---

## Future Improvements

- Add semantic search using embeddings
- Build a streamlit dashboard
- Add source credibility scoring
- Store articles in a vector database for historical search
