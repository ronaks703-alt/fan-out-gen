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
    try:
        parsed = json.loads(re.search(r'\{.*\}', raw, re.S).group())
        return parsed.get("synthetic_queries", [])
    except:
        return []

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="SEO Query Idea Generator", layout="wide")
st.title("🔍 SEO Query Idea Generator")

st.markdown("""
**How it works:**  
1. Pick **Google-Aware AI** for more accurate results.
2. Enter your **Gemini API Key**.
3. Enter your **keyword**.
4. Generate two lists:
   - 🟦 **Google-Based Queries** (from real Google AI Overview, if SerpAPI used)
   - ⚪ **AI Suggestions** (AI-generated ideas)
""")

# Step 1 – Data Source
st.header("Step 1 – Choose Data Source")
data_source = st.radio("Pick your data source", [
    "AI Only (quick, no Google data)",
    "Google-Aware AI (requires SerpAPI key)"
])

# Step 2 – API Key for Gemini
st.header("Step 2 – Enter Gemini API Key")
api_key = st.text_input("Gemini API Key", type="password", value=st.secrets.get("GEMINI_KEY", ""))

# Step 3 – SerpAPI Key (only if Google-Aware AI chosen)
serpapi_key = ""
if data_source == "Google-Aware AI (requires SerpAPI key)":
    serpapi_key = st.text_input("SerpAPI Key", type="password", value=st.secrets.get("SERPAPI_KEY", ""))

# Step 4 – Keyword
st.header("Step 4 – Enter Keyword")
keyword = st.text_input("Your keyword or topic:")

# Generate Button
if st.button("Generate Queries"):
    if not keyword.strip():
        st.error("Please enter a keyword.")
    elif not api_key.strip():
        st.error("Please provide your Gemini API key.")
    else:
        llm_config = configure_llm(api_key)
        if not llm_config:
            st.stop()
        llm_client = LLMClient(llm_config)

        # Try fetching AI Overview if SerpAPI provided
        overview_text, citations, seeds = "", [], []
        if serpapi_key.strip():
            overview_data = fetch_ai_overview(keyword, serpapi_key)
            overview_text = overview_data["text"]
            citations = overview_data["citations"]
            seeds = extract_entities_and_questions(overview_text, llm_client)

        # Generate fan-out queries
        queries = generate_fanout(keyword, "AI_Overview", llm_client, seed_queries=seeds)

        # Mark sources
        df = pd.DataFrame(queries)
        if not df.empty:
            df["source"] = df["query"].apply(lambda q: "Google-Based" if any(q.lower() in s.lower() or s.lower() in q.lower() for s in seeds) else "AI Suggestion")
            
            # Show Google Overview Snapshot if available
            if overview_text:
                with st.expander("📄 Google AI Overview Snapshot"):
                    st.write(overview_text)
                    if citations:
                        st.markdown("**Sources cited by Google:**")
                        for link in citations:
                            st.markdown(f"- [{link}]({link})")

            # Show Google-Based Queries
            google_df = df[df["source"] == "Google-Based"]
            if not google_df.empty:
                st.subheader("🟦 Google-Based Queries")
                st.dataframe(google_df.style.set_properties(**{'background-color': 'lightblue'}))

            # Show AI Suggestions
            ai_df = df[df["source"] == "AI Suggestion"]
            if not ai_df.empty:
                st.subheader("⚪ AI Suggestions")
                st.dataframe(ai_df)

        else:
            st.warning("No queries generated.")
