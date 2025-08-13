import streamlit as st
import google.generativeai as genai
import json, re, time
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional

# Streamlit page setup
st.set_page_config(
    page_title="AI Query Fan-Out Generator",
    page_icon="🔍",
    layout="wide"
)

@dataclass
class SyntheticQuery:
    query: str
    type: str
    user_intent: str
    reasoning: str
    confidence_score: float

# ===== Helper: Robust JSON extraction =====
def safe_json_extract(raw_text: str):
    # Remove markdown fences
    if "```json" in raw_text:
        raw_text = raw_text.split("```json")[1].split("```")[0]
    elif "```" in raw_text:
        raw_text = raw_text.split("```")[1]
    
    # Find JSON object
    match = re.search(r"\{.*\}", raw_text, re.S)
    if not match:
        return None

    json_str = match.group()
    # Remove trailing commas
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

# ===== Core generator class =====
class QueryFanOutGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
    
    def build_prompt(self, keyword: str, mode: str):
        return f"""
You are an SEO keyword expansion assistant.

Primary Keyword: "{keyword}"

Mode: {"AI Overview (10-15 queries)" if mode=="overview" else "AI Mode (15-20 queries)"}

Instructions:
1. Generate diverse related search queries.
2. Cover informational, commercial, transactional, and navigational intents.
3. Output ONLY valid JSON in this format:
{{
  "query_count_reasoning": "why you chose N queries",
  "synthetic_queries": [
    {{
      "query": "text",
      "type": "reformulation|related_query|comparative_query|how_to",
      "user_intent": "informational|commercial|transactional|navigational",
      "reasoning": "short reason",
      "confidence_score": 0.0
    }}
  ]
}}
"""
    
    def generate(self, keyword: str, mode: str):
        prompt = self.build_prompt(keyword, mode)
        try:
            response = self.model.generate_content(prompt)
            parsed = safe_json_extract(response.text)
            if not parsed:
                return None
            queries = [
                SyntheticQuery(
                    query=q["query"],
                    type=q["type"],
                    user_intent=q.get("user_intent", ""),
                    reasoning=q["reasoning"],
                    confidence_score=q.get("confidence_score", 0.0)
                )
                for q in parsed.get("synthetic_queries", [])
            ]
            return {
                "primary_keyword": keyword,
                "query_count_reasoning": parsed.get("query_count_reasoning", ""),
                "synthetic_queries": [asdict(q) for q in queries]
            }
        except Exception as e:
            st.error(f"Error generating queries: {str(e)}")
            return None

# ===== UI =====
st.title("🔍 AI Query Fan-Out Generator (Gemini Only)")
st.markdown("Generate SEO-friendly keyword expansions instantly with AI.")

# API key input
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
if not api_key:
    st.sidebar.info("Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")

# Mode select
mode = st.radio("Choose Generation Mode", ["overview", "full"], format_func=lambda x: "AI Overview" if x=="overview" else "AI Mode")

# Keyword input LAST
keyword = st.text_input("Enter Primary Keyword", placeholder="e.g., Wonderla amusement park")

# Generate button
if st.button("Generate Queries") and api_key and keyword:
    with st.spinner("Generating..."):
        gen = QueryFanOutGenerator(api_key)
        result = gen.generate(keyword, mode)
    
    if result:
        st.subheader(f"Results for: {keyword}")
        st.write(f"**Reasoning:** {result['query_count_reasoning']}")
        
        df = pd.DataFrame(result["synthetic_queries"])
        st.dataframe(df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False),
                file_name=f"{keyword.replace(' ','_')}_queries.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                "📥 Download JSON",
                json.dumps(result, indent=2),
                file_name=f"{keyword.replace(' ','_')}_queries.json",
                mime="application/json"
            )
    else:
        st.warning("No queries generated. Try rephrasing your keyword.")
