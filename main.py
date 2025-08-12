import streamlit as st
import pandas as pd
import json
import re
import serpapi
import openai
import anthropic
import google.generativeai as genai

# ----------------- LLM Provider Setup -----------------
def configure_llm(provider: str, api_key: str, model_name: str):
    if provider == "Gemini":
        genai.configure(api_key=api_key)
        return {"provider": "gemini", "client": genai.GenerativeModel(model_name)}
    elif provider == "OpenAI":
        openai.api_key = api_key
        return {"provider": "openai", "model_name": model_name}
    elif provider == "Claude":
        return {"provider": "claude", "client": anthropic.Anthropic(api_key=api_key), "model_name": model_name}
    else:
        raise ValueError("Unsupported provider")

class LLMClient:
    def __init__(self, config: dict):
        self.config = config

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
        provider = self.config["provider"]
        if provider == "gemini":
            resp = self.config["client"].generate_content(prompt, temperature=temperature, max_output_tokens=max_tokens)
            return getattr(resp, "text", str(resp))
        elif provider == "openai":
            resp = openai.ChatCompletion.create(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        elif provider == "claude":
            resp = self.config["client"].messages.create(
                model=self.config["model_name"],
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text
        else:
            raise ValueError(f"Unknown provider {provider}")

# ----------------- SerpAPI fetch -----------------
def fetch_ai_overview(keyword: str, serpapi_key: str):
    client = serpapi.GoogleSearch({"q": keyword, "engine": "google", "api_key": serpapi_key})
    results = client.get_dict()
    overview = results.get("ai_overview", {})
    return {
        "text": overview.get("text", ""),
        "citations": [c.get("link") for c in overview.get("citations", []) if c.get("link")]
    }

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
Welcome! This tool helps you generate **SEO keyword ideas** in two ways:
- 🟦 **Google-Based** → From actual Google AI Overview (more accurate)
- ⚪ **AI Suggestions** → AI-generated ideas based on your topic
""")

# Step 1 – Topic
st.header("Step 1 – Enter Your Topic")
keyword = st.text_input("Type the topic or keyword you want ideas for:")

# Step 2 – Data Source
st.header("Step 2 – Choose Data Source")
data_source = st.radio("Pick your data source", [
    "AI Only (quick, no Google data)",
    "Google-Aware AI (requires SerpAPI key)"
])

# Step 3 – AI Setup
st.header("Step 3 – Choose AI")
provider = st.selectbox("AI Provider", ["Gemini", "OpenAI", "Claude"])
if provider == "Gemini":
    model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    api_key = st.text_input("Gemini API Key", type="password", value=st.secrets.get("GEMINI_KEY", ""))
elif provider == "OpenAI":
    model_name = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
    api_key = st.text_input("OpenAI API Key", type="password", value=st.secrets.get("OPENAI_KEY", ""))
elif provider == "Claude":
    model_name = st.selectbox("Model", ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"])
    api_key = st.text_input("Anthropic API Key", type="password", value=st.secrets.get("CLAUDE_KEY", ""))

serpapi_key = ""
if data_source == "Google-Aware AI (requires SerpAPI key)":
    serpapi_key = st.text_input("SerpAPI Key", type="password", value=st.secrets.get("SERPAPI_KEY", ""))

mode = st.radio("Search Mode", ["AI_Overview", "AI_Mode"])

# Step 4 – Generate
if st.button("Generate Queries"):
    if not keyword.strip():
        st.error("Please enter a topic or keyword.")
    elif not api_key.strip():
        st.error("Please provide your AI API key.")
    else:
        llm_client = LLMClient(configure_llm(provider, api_key, model_name))

        # Try fetching AI Overview if SerpAPI provided
        overview_text, citations, seeds = "", [], []
        if serpapi_key.strip():
            overview_data = fetch_ai_overview(keyword, serpapi_key)
            overview_text = overview_data["text"]
            citations = overview_data["citations"]
            seeds = extract_entities_and_questions(overview_text, llm_client)

        # Generate fan-out queries
        queries = generate_fanout(keyword, mode, llm_client, seed_queries=seeds)

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
