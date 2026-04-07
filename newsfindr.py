"""
NewsFindr: LLM-powered News Retrieval System

Pipeline:
User Query → Query Expansion → Web Search → Credibility Filtering → Summarization
"""
# =============================
# Imports
# =============================
import json
import os
import pandas as pd
import sqlite3

from langchain.agents import create_sql_agent, initialize_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langchain.agents import load_tools
from langchain.agents import Tool

from pydantic import BaseModel, Field, ValidationError

from ddgs import DDGS
from typing import List, Optional, Dict

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_groq import ChatGroq

# =============================
# LLM Configuration
# =============================
# Get the API key from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')

# Low creativity (deterministic) LLM for consistent responses
llm = ChatGroq(
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0.2,
    groq_api_key=groq_api_key,
    max_retries=2
)
# =============================
# SQL Agent for Data Retrieval
# =============================

db = SQLDatabase.from_uri("sqlite:///customer.db")

system_message_sql = ("""
You are an SQL expert working with a SQLite database.

The database contains a table named customers.

Rules:
- You MUST use the provided SQL tools to answer all questions.
- ALWAYS query the customers table.
- NEVER hallucinate tables or columns.
- Do NOT explain your reasoning or steps.
- Only return the final answer based on SQL query results.

When an email is asked:
- Use the SQL tool to query the customers table using the email column.
- If the email does NOT exist, reply exactly: email not found
- If the email exist, return ONLY a Python list of strings representing the customer's interests, for example: ['Politics', 'Startups']. Do NOT include the customer's name, any explanations, thoughts, or tool invocations in the final response. Just the Python list.

When asked for unique emails:
- Execute the SQL query to get distinct emails from the 'customers' table.
- Your final response must ONLY be a Python list of strings representing the unique emails, for example: ['email1@example.com', 'email2@example.com']. Ensure that the output is a flat list of strings, not a list of tuples or any other format. Do NOT include any explanations, thoughts, or tool invocations in the final response. Just the Python list.

Output rules:
- Do NOT explain reasoning.
- Do NOT mention tools, errors, or iteration limits.
- Your final response must be the direct answer, and ONLY the direct answer.
"""    )

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

db_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=False,
    system_message=SystemMessage(system_message_sql),
    handle_parsing_errors=True,
    max_iteration=3
)

# =====================================================
# Generating Expanded Search Queries for Current News
# =====================================================
def expand_search_queries(inputs) -> list:
    """
    Expand user interests into time-sensitive search queries for breaking news.
    Accepts either a dict (with 'interests' and 'user_query') or a raw string.
    """
    # Handle dict input
    if isinstance(inputs, dict):
        interests = inputs.get("interests", [])
        user_query = inputs.get("user_query","")
    else:
        # Handle string input fallback
        interests = [i.strip() for i in str(inputs).split(",")]
        user_query = ""

    system_prompt_expand = """You are a news search query generator. Your task is to transform user interests into precise and up-to-date search queries for current news. Focus on creating queries that will yield breaking news, trending topics, or recent developments. Avoid including specific years unless explicitly requested, as the goal is to find the latest information. Ensure the queries are suitable for a search engine like DuckDuckGo."""

    expanded_queries = []
    for interest in interests:
        prompt = f"Generate one search query related to: '{interest}' considering the user query: '{user_query}'"
        response = llm.predict_messages(
            [
                SystemMessage(content=system_prompt_expand),
                HumanMessage(content=prompt)
            ]
        )
        query = response.content.strip()
        if query:
            expanded_queries.append(query)

    return expanded_queries

expand_tool = Tool(
    name="ExpandSearchQueries",
    func=expand_search_queries,
    description="Expands user interests into precise, time-sensitive news search queries based on a user query."
)
# =============================
# Fetch News Results Using DuckDuckGo 
# =============================

def ddg_search(query: str) -> str:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            title = r.get("title", "")
            url = r.get("href", "")
            body = r.get("body", "")
            res={"title":title,"url":url,"body":body}
            results.append(res)
    return results
ddg_search_tool = Tool(
    name="DuckDuckGoSearch",
    func=ddg_search,
    description="Searches DuckDuckGo for recent news and returns top 5 results with URLs."
)

# ====================================================================
#  Filter Relevant and Trustworthy URLs Based on User Interests
# ====================================================================

def filter_with_llm(search_results: list) -> list:

    search_results= json.dumps(search_results, indent=2)
    system_prompt_filter = """You are a highly skilled AI assistant tasked with evaluating the credibility and relevance of news articles based on provided search results. Your goal is to filter out irrelevant or untrustworthy sources, and return only the URLs of articles that are highly relevant to the user's interests and come from reputable sources. Consider factors like the domain name, the presence of 'ads' or 'sponsored' content indicators in the URL or description, and general web reputation. Output only a JSON array of credible URLs, without any additional text or explanations."""

    prompt = "Evaluate the credibility and relevance of the following search results and return only a JSON array of the credible URLs:\n" + search_results

    response = llm.predict_messages(
        [
            SystemMessage(content=system_prompt_filter),
            HumanMessage(content=prompt)
        ]
    )
    return response.content


credibility_tool = Tool(
    name="CredibilityFilter",
    func=filter_with_llm,
    description="Filters search results by evaluating URL credibility using LLM."
)

# =============================
# Generate summary for the URLs 
# =============================

def summarize_news(url_list) -> dict:
    """
    Summarizes key news points from a list of URLs and returns both the summary and the URLs.
    Since only URLs are available, the LLM will generate a high-level summary
    based on the website context or domain.
    """

    system_prompt_summarize = """You are an AI assistant designed to summarize news articles. Given a list of URLs, your task is to provide a concise and informative summary of the key points from each article. If content cannot be directly accessed, provide a high-level summary based on the domain or available metadata. The summary should be neutral and objective."""

    # Pass the URLs to the LLM for summarization
    prompt = "Summarize the key news points from the following URLs:\n" + "\n".join(url_list)

    response = llm.predict_messages(
        [
            SystemMessage(content=system_prompt_summarize),
            HumanMessage(content=prompt)
        ]
    )

    summary_text = response.content

    # Return a dictionary containing the summary and the original list of URLs
    return {"summary": summary_text, "urls": url_list}


summarize_tool = Tool(
    name="SummarizeNews",
    func=summarize_news,
    description="Generates summaries from news URLs and returns both the summary and the links."
)


# =============================
# Creating the Main Agent 
# =============================


tools = [expand_tool, ddg_search_tool, credibility_tool, summarize_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# =============================
# Main Query Response Function
# =============================

def query_response(email: str, user_query: str) -> str:
    # Fetch interests based on the provided email
    interest_result = db_agent.invoke({"input": f"Fetch all the Interests with email_id '{email}' in a list"})

    # Normalize to list
    if isinstance(interest_result["output"], dict) and "items" in interest_result["output"]:
        interests = interest_result["output"]["items"]
    elif isinstance(interest_result["output"], str):
        interests = [i.strip() for i in interest_result["output"].split(",")]
    else:
        interests = interest_result["output"]

    # Agent prompt
    agent_prompt = f"""
    The user's interest is: {interests} and specifically looking for {user_query}.
    Here is the process to follow:
    1. Expand the user's interest into news search queries using the 'ExpandSearchQueries' tool.
       The input to this tool should be a dictionary like: {{'interests': {interests}, 'user_query': '{user_query}'}}.
    2. Use the query generated in step 1 to search for recent news using the 'DuckDuckGoSearch' tool for all the queries.
    3. Pass the combined results of all queries into 'CredibilityFilter' for filtering.
    4. From the filtered results, select the top 3 latest news articles for summarization.
    5. Summarize all credible results into a detailed summary using the 'SummarizeNews'.
    6. Provide the final summary to the user with credible source urls.
    """
    response = agent.run(agent_prompt)
    print("\n======= FINAL RESPONSE =======")
    print(response)

    return response

# =============================
# Example Usage 
# =============================

if __name__ == "__main__":

    # Example 
    email = "test@example.com"
    user_query = "recent political and startup news"
    response = query_response(email, user_query)
