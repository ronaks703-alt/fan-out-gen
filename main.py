import streamlit as st
import pandas as pd
import json
import re
import serpapi
import google.generativeai as genai

# ----------------- LLM Provider Setup -----------------
def configure_llm(api_key: str):
    try:
        genai.configure(api_key=api_key)
        return {"provider": "gemini", "client": genai.GenerativeModel("gemini-1.5-flash")}
    except Exception as e:
        st.error(f"Gemini configuration failed: {e}")
        return None

class LLMClient:
    def __init__(self, config: dict):
        self.config = config

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
        provider = self.config["provider"]
        try:
            if provider == "gemini":
                resp = self.config["client"].generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens
                    }
                )
                # Log the response from Gemini to debug the output
                st.write("Gemini Response:", resp.text)  # This will print the response to the Streamlit UI for debugging
                return getattr(resp, "text", str(resp))
        except Exception as e:
            st.error(f"Error generating with {provider}: {e}")
            return ""

# ----------------- SerpAPI fetch -----------------
def fetch_ai_overview(keyword: str, serpapi_key: str):
    try:
        client = serpapi.GoogleSearch({"q": keyword, "engine": "google", "api_key": serpapi_key})
        results = client.get_dict()
        overview = results.get("ai_overview", {})
        return {
            "text": overview.get("text", ""),
            "citations": [c.get("link") for c in overview.get("citations", []) if c.get("link")]
        }
    except Exception as e:
        st.warning(f"SerpAPI fetch failed: {e}")
        return {"text": "", "citations": []}

# ----------------- Entity & question extraction -----------------
def extract_entities_and_questions(overview_text: str, llm: LLMClient):
    if not overview_text.strip():
        return []
    prompt = f"""
    You are given Google's AI Overview text:
    \"\"\"{overview_text}\"\"\"
    Extract a list of up to 10 important search queries from it, including:
    - Questions mentioned or implied
    - Entities, names, or topics worth searching
    Return JSON with a 'queries' array, no extra text.
    """
    raw = llm.generate(prompt)
    try:
        parsed = json.loads(re.search(r'\{.*\}', raw, re.S).group())
        return parsed.get("queries", [])
    except:
        return []

# ----------------- Fan-out generation -----------------
def generate_fanout(primary_keyword: str, search_mode: str, llm: LLMClient, seed_queries=None):
    seed_text = ""
    if seed_queries:
        seed_text = "\nThese are seed queries from Google's AI Overview:\n" + "\n".join(f"- {q}" for q in seed_queries)
    target_range = "10-15" if search_mode == "AI_Overview" else "15-20"
    prompt = f"""
    Generate {target_range} diverse, high-quality search queries for the topic "{primary_keyword}".
    Include a mix of:
    - Reformulations
    - Related queries
    - Comparisons
    - How-to's
    For each query, also return:
    - type (reformulation, related_query, comparative_query, how_to, etc.)
    - user_intent (informational, commercial, transactional, navigational)
    - reasoning
    - confidence_score (0-1)

    Output as JSON:
    {{
      "synthetic_queries": [
        {{
          "query": "...",
          "type": "...",
          "user_intent": "...",
          "reasoning": "...",
          "confidence_score": 0.85
        }}
      ]
    }}
    {seed_text}
    """
    raw = llm.generate(prompt)
    # Log the raw response to check if queries are being returned
    st.write("Raw Response from LLM:", raw)  # This will show the raw data for debugging purposes
    try:
        parsed = json.loads(re.search(r'\{.*\}', raw, re.S).group())
        return parsed.get("synthetic_queries", [])
    except Exception as e:
        st.error(f"Error parsing LLM response: {e}")
        return []

# ----------------- Streamlit UI -----------------
st.set_page_config(page
