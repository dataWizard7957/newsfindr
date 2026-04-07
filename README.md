# NewsFindr – LLM Powered Personalized News Agent

AI-powered system that retrieves personalized and credible news articles using an LLM-based agent with query expansion, web search, credibility filtering, and automated summarization.

## Problem

Searching for relevant and reliable news is challenging because:

- Users receive generic results not tailored to their interests
- Search engines often return duplicate or low-credibility sources
- Important information is spread across multiple articles
- Users must manually read several articles to understand the overall story
- An intelligent system is needed to personalize, filter, and summarize news automatically.

## Approach

The system implements an LLM-powered AI agent pipeline:

- Personalization: Fetches user interests using email from a database
- Query Expansion: Uses an LLM to generate multiple related search queries
- Web Search: Retrieves news articles using DuckDuckGo
- Credibility Filtering: Filters irrelevant or low-quality sources
- Summarization: Generates a concise summary of key news insights

## Pipeline
```text
User Email + Query
        ↓
Fetch User Interests (Database)
        ↓
AI Agent (Tool Calling + Reasoning)
        ↓
Query Expansion
        ↓
Web Search (DuckDuckGo)
        ↓
Credibility Filtering
        ↓
News Summarization
        ↓
Personalized News Insights
```

## Results
- Personalized news retrieval based on user interests
- Improved search coverage using LLM query expansion
- Reduced noise through filtering
- Generated concise summaries from multiple sources
- Demonstrates a practical AI agent-based retrieval system

## Tech Stack
- Python
- Groq API (LLM)
- LangChain (Agents + Tools)
- DuckDuckGo Search
- SQLite

## Project Structure
```text
newsfindr/
│
├── newsfindr.py
├── customer.db
├── requirements.txt
├── README.md
```

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
### Example
- Input:
User email and query
- Output:
List of relevant news articles
AI-generated summary
Structured response

```bash
{
  "summary": "...",
  "urls": [...]
}
```

## Limitations
- Summaries are generated from search results (metadata/snippets), not full article content
- Credibility filtering is heuristic and may not always be accurate
- Depends on DuckDuckGo search results

## Future Improvements
- Add full article extraction (e.g., newspaper3k)
- Build a streamlit dashboard
- Implement source credibility scoring
