import streamlit as st
import google.generativeai as genai
import requests
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass, asdict
import re

# Configure Streamlit page
st.set_page_config(
    page_title="Query Fan-Out Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Session state setup ----
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_serp_results' not in st.session_state:
    st.session_state.last_serp_results = None


# ---- Dataclasses ----
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


# ---- Query Generator ----
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

    def _build_cot_prompt(self, primary_keyword: str, search_mode: str,
                          serp_context: Optional[str] = None,
                          industry_context: Optional[str] = None,
                          query_intent: Optional[str] = None,
                          user_persona: Optional[str] = None) -> str:

        repeated = " ".join([primary_keyword] * 5)
        lines = [f'Repeated Query Bias: "{repeated}"',
                 f'Answer the following query: "{primary_keyword}"',
                 "Give your reasoning step by step—explain how each distinct angle, "
                 "facet, or subtopic is discovered."]

        if industry_context:
            lines.append(f"Industry Context: {industry_context}")
        if query_intent:
            guidance = {
                "informational": "Focus on definitions, background, and explanatory facets.",
                "commercial": "Emphasize price, comparison, and product/vendor alternatives.",
                "transactional": "Generate queries about where/how to buy, availability, pricing.",
                "navigational": "Create brand/site-specific or service-specific expansions."
            }.get(query_intent, "")
            lines.append(f"Query Intent: {query_intent} — {guidance}")
        if user_persona:
            lines.append(f"User Persona: {user_persona}")
        if serp_context:
            lines.append("SERP Context (from top search results):")
            lines.append(serp_context)

        if search_mode == "AI_Mode":
            lines.append("Generate 15–20 complex and diverse queries that cover all relevant facets.")
        else:
            lines.append("Generate 10–15 focused queries to provide a comprehensive AI Overview.")

        lines.append(
            "Finally, output valid JSON in exactly this format:\n"
            '''{
  "query_count_reasoning": "Your explanation of why you chose exactly N queries",
  "synthetic_queries": [
    {
      "query": "…",
      "type": "one of: reformulation|related_query|implicit_query|comparative_query|entity_expansion|personalized_query",
      "user_intent": "…",
      "reasoning": "…",
      "confidence_score": 0.0–1.0
    }
  ]
}'''
        )

        return "\n\n".join(lines)

    def _filter_queries(self, primary_keyword: str, queries: List[SyntheticQuery]) -> List[SyntheticQuery]:
        filtered = []
        primary_words = set(primary_keyword.lower().split())
        for query in queries:
            words = query.query.strip().split()
            if len(words) < 2 or len(words) > 15:
                continue
            if set(query.query.lower().split()) == primary_words:
                continue
            if not re.search(r'[a-zA-Z]', query.query):
                continue
            duplicate = False
            for existing in filtered:
                w1, w2 = set(query.query.lower().split()), set(existing.query.lower().split())
                sim = len(w1 & w2) / len(w1 | w2)
                if sim > 0.7:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(query)
        return filtered

    def generate_fanout(self, primary_keyword: str, search_mode: str,
                        serp_context: Optional[str] = None,
                        industry_context: Optional[str] = None,
                        query_intent: Optional[str] = None,
                        user_persona: Optional[str] = None) -> Optional[Dict]:
        prompt = self._build_cot_prompt(primary_keyword, search_mode,
                                        serp_context, industry_context, query_intent, user_persona)
        start_time = time.time()
        try:
            response = self.model.generate_content(prompt)
            generation_time_ms = int((time.time() - start_time) * 1000)
            response_text = response.text
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1]

            json_match = re.search(r'\{.*\}', response_text.strip(), re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in model response")
            response_data = json.loads(json_match.group())

            synthetic_queries = []
            for sq in response_data.get('synthetic_queries', []):
                synthetic_queries.append(SyntheticQuery(
                    query=sq['query'],
                    type=sq['type'],
                    user_intent=sq.get('user_intent', ""),
                    reasoning=sq['reasoning'],
                    confidence_score=float(sq.get('confidence_score', 0.8))
                ))

            filtered = self._filter_queries(primary_keyword, synthetic_queries)

            result = {
                'primary_keyword': primary_keyword,
                'query_count_reasoning': response_data.get('query_count_reasoning', ""),
                'synthetic_queries': [asdict(q) for q in filtered],
                'generation_metadata': asdict(GenerationMetadata(
                    total_queries_generated=len(filtered),
                    generation_time_ms=generation_time_ms,
                    model_used='gemini-2.0-flash',
                    prompt_version=self.prompt_version
                ))
            }
            return result
        except Exception as e:
            st.error(f"Error generating queries: {str(e)}")
            return None


# ---- SERP Helper ----
def fetch_serp_results(api_key: str, query: str, num_results: int = 10) -> List[Dict]:
    try:
        url = "https://serpapi.com/search.json"
        params = {"q": query, "hl": "en", "gl": "us", "api_key": api_key}
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        results = []
        for i, item in enumerate(data.get("organic_results", [])[:num_results]):
            results.append({
                "rank": i + 1,
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet", "")
            })
        return results
    except Exception as e:
        st.error(f"Error fetching SERP results: {e}")
        return []


# ---- UI ----
def main():
    st.title("🔍 Query Fan-Out Generator")

    st.markdown("""
    Generate synthetic queries from primary keywords using Chain-of-Thought prompting,
    mimicking Google's AI Overview algorithm.

    **Modes:**  
    - **AI Overview** 🛈: 10–15 focused queries (like Google's AI Overviews summary).  
    - **AI Mode** 🛈: 15–20 broader, complex queries covering multiple angles.  
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input("Gemini API Key", type="password")
        serp_api_key = st.text_input("SERP API Key (Optional)", type="password")
        include_serp = st.checkbox("Include SERP Results", value=False,
                                   help="If ON and SERP API key is provided, top search results will be fetched and merged into the prompt.")

        search_mode = st.selectbox("Search Mode", ["AI_Overview", "AI_Mode"])
        query_intent = st.selectbox("Query Intent (Optional)", ["", "informational", "commercial", "transactional", "navigational"])
        industry_context = st.text_input("Industry Context (Optional)")
        user_persona = st.text_area("User Persona (Optional)")

    primary_keyword = st.text_input("Primary Keyword", placeholder="Enter your keyword")
    if st.button("Generate Fan-Out", type="primary") and api_key and primary_keyword:
        serp_context = None
        serp_results = None
        if serp_api_key and include_serp:
            serp_results = fetch_serp_results(serp_api_key, primary_keyword)
            if serp_results:
                serp_context = "\n".join([f"{r['title']} — {r['snippet']}" for r in serp_results])

        generator = QueryFanOutGenerator(api_key)
        result = generator.generate_fanout(primary_keyword, search_mode,
                                           serp_context=serp_context,
                                           industry_context=industry_context or None,
                                           query_intent=query_intent or None,
                                           user_persona=user_persona or None)
        if result:
            st.session_state.last_result = result
            st.session_state.last_serp_results = serp_results

    # ---- Display last results ----
    if st.session_state.last_result:
        result = st.session_state.last_result
        st.markdown("## 🧠 Generated Queries")
        queries_df = pd.DataFrame(result['synthetic_queries'])
        st.dataframe(queries_df, use_container_width=True, height=500)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download CSV", queries_df.to_csv(index=False),
                               file_name=f"fanout_{result['primary_keyword']}.csv", mime="text/csv")
        with col2:
            st.download_button("📥 Download JSON", json.dumps(result, indent=2),
                               file_name=f"fanout_{result['primary_keyword']}.json", mime="application/json")

    if st.session_state.last_serp_results and include_serp:
        serp_results = st.session_state.last_serp_results
        with st.expander("🔎 SERP Results (Top 10)"):
            serp_df = pd.DataFrame(serp_results)
            st.dataframe(serp_df, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 SERP CSV", serp_df.to_csv(index=False),
                                   file_name="serp_results.csv", mime="text/csv")
            with col2:
                st.download_button("📥 SERP JSON", json.dumps(serp_results, indent=2),
                                   file_name="serp_results.json", mime="application/json")


if __name__ == "__main__":
    main()
