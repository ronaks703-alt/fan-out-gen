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

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Query Fan-Out Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- SESSION STATE ----------
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'serp_results' not in st.session_state:
    st.session_state.serp_results = None

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

# ---------- GEMINI GENERATOR ----------
class QueryFanOutGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.prompt_version = "1.1.0"

    def _build_prompt(self, primary_keyword, search_mode, industry_context=None,
                      query_intent=None, user_persona=None, serp_context=None):
        repeated = " ".join([primary_keyword] * 5)
        lines = [f"Repeated Query Bias: '{repeated}'"]
        lines.append(f"Primary Keyword: {primary_keyword}")
        lines.append("Explain reasoning for each generated query.")
        if industry_context:
            lines.append(f"Industry Context: {industry_context}")
        if query_intent:
            lines.append(f"Query Intent: {query_intent}")
        if user_persona:
            lines.append(f"User Persona: {user_persona}")
        if search_mode == "AI_Overview":
            lines.append("Generate 10–15 queries to give a broad overview.")
        else:
            lines.append("Generate 15–20 complex, diverse queries covering all facets.")
        if serp_context:
            lines.append("Use the following search results for additional context:\n" + serp_context)
        lines.append("""Output valid JSON:
{
  "query_count_reasoning": "...",
  "synthetic_queries": [
    {
      "query": "...",
      "type": "reformulation|related_query|comparative_query|how_to|transactional",
      "user_intent": "...",
      "reasoning": "...",
      "confidence_score": 0.0–1.0
    }
  ]
}""")
        return "\n".join(lines)

    def _filter_queries(self, primary_keyword, queries: List[SyntheticQuery]):
        filtered = []
        for q in queries:
            if len(q.query.strip()) < 3:
                continue
            if q.query.lower() == primary_keyword.lower():
                continue
            if not any(ch.isalpha() for ch in q.query):
                continue
            if any(self._similarity(q.query, fq.query) > 0.7 for fq in filtered):
                continue
            filtered.append(q)
        return filtered

    def _similarity(self, q1, q2):
        w1, w2 = set(q1.lower().split()), set(q2.lower().split())
        return len(w1 & w2) / len(w1 | w2) if w1 and w2 else 0.0

    def generate_fanout(self, primary_keyword, search_mode, industry_context=None,
                        query_intent=None, user_persona=None, serp_context=None):
        prompt = self._build_prompt(primary_keyword, search_mode,
                                    industry_context, query_intent,
                                    user_persona, serp_context)
        start_time = time.time()
        try:
            resp = self.model.generate_content(prompt)
            elapsed = int((time.time() - start_time) * 1000)
            text = resp.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON in model output")
            data = json.loads(json_match.group())
            queries = [SyntheticQuery(**sq) for sq in data.get("synthetic_queries", [])]
            filtered = self._filter_queries(primary_keyword, queries)
            return {
                "primary_keyword": primary_keyword,
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
            st.error(f"Error generating queries: {e}")
            return None

# ---------- SERP FETCH ----------
def fetch_serp_results(api_key, query):
    try:
        url = f"https://serpapi.com/search.json?q={query}&engine=google&api_key={api_key}"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        organic = data.get("organic_results", [])
        st.session_state.serp_results = organic
        serp_context = "\n".join(
            f"- {item.get('title')} ({item.get('link')}): {item.get('snippet', '')}"
            for item in organic
        )
        return serp_context
    except Exception as e:
        st.error(f"SERP API error: {e}")
        return None

# ---------- UI ----------
def main():
    st.title("🔍 Query Fan-Out Generator")

    with st.sidebar:
        api_key = st.text_input("Gemini API Key", type="password")
        serp_api_key = st.text_input("SERP API Key (Optional)", type="password")
        include_serp = st.checkbox("Include SERP Results", value=False, help="If ON and SERP API key provided, search results will be fetched and used as extra context.")
        search_mode = st.selectbox("Search Mode", ["AI_Overview", "AI_Mode"],
                                   help="AI Overview: Broad coverage (10–15 queries)\nAI Mode: Deep dive (15–20 queries)")
        query_intent = st.selectbox("Query Intent (Optional)", ["", "informational", "commercial", "transactional", "navigational"])
        industry_context = st.text_input("Industry Context (Optional)")
        user_persona = st.text_area("User Persona (Optional)")

    keyword = st.text_input("Primary Keyword")
    if st.button("Generate") and api_key and keyword:
        serp_context = None
        if include_serp and serp_api_key:
            serp_context = fetch_serp_results(serp_api_key, keyword)
            if st.session_state.serp_results:
                with st.expander("🔎 SERP Results"):
                    df = pd.DataFrame(st.session_state.serp_results)
                    st.dataframe(df[["title", "link", "snippet"]], use_container_width=True)
                    st.download_button("Download SERP CSV", df.to_csv(index=False), "serp_results.csv", "text/csv")
                    st.download_button("Download SERP JSON", json.dumps(st.session_state.serp_results, indent=2), "serp_results.json", "application/json")
        gen = QueryFanOutGenerator(api_key)
        result = gen.generate_fanout(keyword, search_mode, industry_context or None, query_intent or None, user_persona or None, serp_context)
        if result:
            df = pd.DataFrame(result["synthetic_queries"])
            st.subheader("Generated Queries")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), f"{keyword}_queries.csv", "text/csv")
            st.download_button("Download JSON", json.dumps(result, indent=2), f"{keyword}_queries.json", "application/json")

if __name__ == "__main__":
    main()
