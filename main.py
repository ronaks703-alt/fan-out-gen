import streamlit as st
import google.generativeai as genai
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass, asdict
import re
import requests

# ----------------- STREAMLIT PAGE CONFIG -----------------
st.set_page_config(
    page_title="Query Fan-Out Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- SESSION STATE INIT -----------------
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []

# ----------------- DATA CLASSES -----------------
@dataclass
class GenerationMetadata:
    total_queries_generated: int
    generation_time_ms: int
    model_used: str
    prompt_version: str

@dataclass
class SyntheticQuery:
    query: str
    type: str
    user_intent: str
    reasoning: str
    confidence_score: float

# ----------------- SERP API HELPER -----------------
def fetch_serp_results(keyword: str, serp_api_key: str, num_results: int = 10):
    """Fetch top Google search results via SERP API."""
    url = "https://serpapi.com/search"
    params = {
        "q": keyword,
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": serp_api_key,
        "num": num_results
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        st.error(f"SERP API error: {resp.status_code}")
        return []
    data = resp.json()
    results = []
    for idx, item in enumerate(data.get("organic_results", []), start=1):
        results.append({
            "position": idx,
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
            "favicon": item.get("favicon", "")
        })
    return results

# ----------------- QUERY GENERATOR -----------------
class QueryFanOutGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        self.model = genai.GenerativeModel('gemini-2.0-flash', safety_settings=safety_settings)
        self.prompt_version = "1.1.0"

    def _build_prompt(self, keyword: str, mode: str, serp_results: Optional[List[Dict]] = None):
        prompt_parts = [
            f"Primary Keyword: {keyword}",
            "Generate search queries using chain-of-thought reasoning.",
            f"Mode: {mode} — {'10–15 focused queries for concise AI overviews' if mode == 'AI_Overview' else '15–20 diverse queries covering multiple aspects'}",
        ]
        if serp_results:
            serp_text = "\n".join([f"{r['position']}. {r['title']} — {r['snippet']}" for r in serp_results])
            prompt_parts.append("Here are the current top Google results to consider:\n" + serp_text)
        prompt_parts.append(
            """Output valid JSON:
{
  "query_count_reasoning": "Why you chose N queries",
  "synthetic_queries": [
    {
      "query": "...",
      "type": "reformulation|related_query|implicit_query|comparative_query|entity_expansion|personalized_query",
      "user_intent": "...",
      "reasoning": "...",
      "confidence_score": 0.0–1.0
    }
  ]
}"""
        )
        return "\n\n".join(prompt_parts)

    def _validate_query(self, query: str) -> bool:
        if len(query.split()) < 2 or len(query) < 5:
            return False
        if not re.search(r'[a-zA-Z]', query):
            return False
        return True

    def _dedupe_queries(self, queries: List[SyntheticQuery], keyword: str) -> List[SyntheticQuery]:
        seen = set()
        filtered = []
        for q in queries:
            if self._validate_query(q.query) and q.query.lower() != keyword.lower() and q.query.lower() not in seen:
                seen.add(q.query.lower())
                filtered.append(q)
        return filtered

    def generate(self, keyword: str, mode: str, serp_results: Optional[List[Dict]] = None):
        prompt = self._build_prompt(keyword, mode, serp_results)
        try:
            start = time.time()
            resp = self.model.generate_content(prompt)
            elapsed = int((time.time() - start) * 1000)

            text = resp.text
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                raise ValueError("No valid JSON found.")
            data = json.loads(match.group())

            queries = [SyntheticQuery(**q) for q in data.get("synthetic_queries", [])]
            filtered = self._dedupe_queries(queries, keyword)

            return {
                "primary_keyword": keyword,
                "query_count_reasoning": data.get("query_count_reasoning", ""),
                "synthetic_queries": [asdict(q) for q in filtered],
                "generation_metadata": asdict(GenerationMetadata(
                    total_queries_generated=len(filtered),
                    generation_time_ms=elapsed,
                    model_used="gemini-2.0-flash",
                    prompt_version=self.prompt_version
                ))
            }
        except Exception as e:
            st.error(f"Error: {e}")
            return None

# ----------------- STREAMLIT UI -----------------
def main():
    st.title("🔍 Query Fan-Out Generator")
    st.markdown("""
Generate **synthetic search queries** from a primary keyword using Google's Gemini,  
with optional SERP API integration to align with **current Google rankings**.
""")

    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("Gemini API Key", type="password")
        serp_key = st.text_input("SERP API Key (Optional)", type="password")
        mode = st.selectbox(
            "Search Mode",
            ["AI_Overview", "AI_Mode"],
            help="**AI Overview**: 10–15 focused queries for concise summaries.\n**AI Mode**: 15–20 diverse queries covering all facets."
        )
        include_serp = st.checkbox("Include SERP results in generation", value=False)

    keyword = st.text_input("Primary Keyword", placeholder="e.g., Wonderla amusement park")

    if st.button("Generate Queries") and api_key and keyword:
        serp_results = None
        if include_serp and serp_key:
            with st.spinner("Fetching SERP data..."):
                serp_results = fetch_serp_results(keyword, serp_key)
                if serp_results:
                    with st.expander("🔎 View Top Google Results"):
                        df = pd.DataFrame(serp_results)
                        st.dataframe(df, use_container_width=True)

        with st.spinner("Generating queries with Gemini..."):
            gen = QueryFanOutGenerator(api_key)
            result = gen.generate(keyword, mode, serp_results)

            if result:
                st.success(f"Generated {len(result['synthetic_queries'])} queries.")
                df = pd.DataFrame(result["synthetic_queries"])
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "Download JSON",
                    data=json.dumps(result, indent=2),
                    file_name=f"{keyword.replace(' ', '_')}_queries.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
