"""
NewsFindr: LLM-powered News Retrieval System

Pipeline:
User Query → Query Expansion → Web Search → Credibility Filtering → Summarization
"""

# =============================
# Imports
# =============================

import os
import json
import pandas as pd
from typing import List

from langchain_groq import ChatGroq
from duckduckgo_search import DDGS


# =============================
# LLM Configuration
# =============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY
)


# =============================
# Query Expansion
# =============================

def expand_search_queries(query: str) -> List[str]:
    """
    Expands a user query into multiple search queries
    using the LLM for better retrieval coverage.
    """

    prompt = f"""
    Generate 5 diverse search queries related to:
    {query}
    """

    response = llm.invoke(prompt)

    queries = response.content.split("\n")
    queries = [q.strip("- ").strip() for q in queries if q.strip()]

    return queries


# =============================
# DuckDuckGo Search
# =============================

def ddg_search(queries: List[str], max_results=5):
    """
    Performs web search using DuckDuckGo
    and returns collected news articles.
    """

    results = []

    with DDGS() as ddgs:
        for q in queries:
            search_results = ddgs.text(q, max_results=max_results)

            for r in search_results:
                results.append({
                    "title": r.get("title"),
                    "link": r.get("href"),
                    "snippet": r.get("body")
                })

    return results


# =============================
# Credibility Filtering
# =============================

def filter_with_llm(results):
    """
    Filters low credibility or irrelevant news
    using the LLM.
    """

    filtered = []

    for r in results:

        prompt = f"""
        Determine if the following news source is credible.

        Title: {r['title']}
        Snippet: {r['snippet']}

        Answer only YES or NO.
        """

        response = llm.invoke(prompt)

        if "YES" in response.content.upper():
            filtered.append(r)

    return filtered


# =============================
# News Summarization
# =============================

def summarize_news(results):
    """
    Summarizes the final news articles
    into a concise overview.
    """

    text = "\n".join([r["title"] + " - " + r["snippet"] for r in results])

    prompt = f"""
    Summarize the following news articles.

    {text}
    """

    response = llm.invoke(prompt)

    return response.content


# =============================
# Main Pipeline
# =============================

def query_response(email: str, query: str):
    """
    Complete NewsFindr pipeline.
    """

    print("Expanding query...")
    expanded_queries = expand_search_queries(query)

    print("Searching news...")
    search_results = ddg_search(expanded_queries)

    print("Filtering credible news...")
    filtered_results = filter_with_llm(search_results)

    print("Summarizing news...")
    summary = summarize_news(filtered_results)

    return {
        "email": email,
        "query": query,
        "summary": summary,
        "articles": filtered_results
    }


# =============================
# Example Run
# =============================

if __name__ == "__main__":

    email = "test@example.com"
    query = "latest developments in artificial intelligence"

    result = query_response(email, query)

    print("\nSummary:\n")
    print(result["summary"])
