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

# ----------------- UI -----------------
st.set_page_config(page_title="Query Fan-Out Generator", layout="wide")

st.sidebar.markdown("## LLM Configuration")
provider = st.selectbox("LLM Provider", ["Gemini", "OpenAI", "Claude"])
if provider == "Gemini":
    model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    api_key = st.text_input("Gemini API Key", type="password", value=st.secrets.get("GEMINI_KEY", ""))
elif provider == "OpenAI":
    model_name = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
    api_key = st.text_input("OpenAI API Key", type="password", value=st.secrets.get("OPENAI_KEY", ""))
elif provider == "Claude":
    model_name = st.selectbox("Model", ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"])
    api_key = st.text_input("Anthropic API Key", type="password", value=st.secrets.get("CLAUDE_KEY", ""))

serpapi_key = st.text_input("SerpAPI Key", type="password", value=st.secrets.get("SERPAPI_KEY", ""))

st.title("🔍 Query Fan-Out Generator with AI Overview Seeding")

keyword = st.text_input("Enter primary keyword:")
mode = st.radio("Search Mode", ["AI_Overview", "AI_Mode"])

if st.button("Generate Fan-Out"):
    if not all([api_key, serpapi_key, keyword.strip()]):
        st.error("Please provide API keys and a keyword.")
    else:
        llm_client = LLMClient(configure_llm(provider, api_key, model_name))

        # Step 1: Fetch AI Overview
        overview_data = fetch_ai_overview(keyword, serpapi_key)
        overview_text = overview_data["text"]
        citations = overview_data["citations"]

        if overview_text:
            st.subheader("📄 Google AI Overview Snapshot")
            st.write(overview_text)
        else:
            st.warning("No AI Overview found for this query.")

        if citations:
            st.subheader("🌐 Cited Sources")
            for link in citations:
                st.markdown(f"- [{link}]({link})")

        # Step 2: Extract seed queries
        seeds = extract_entities_and_questions(overview_text, llm_client)
        if seeds:
            st.subheader("🔍 Seed Queries from AI Overview")
            st.write(seeds)

        # Step 3: Generate final fan-out
        queries = generate_fanout(keyword, mode, llm_client, seed_queries=seeds)

        # Step 4: Mark sources
        df = pd.DataFrame(queries)
        if not df.empty:
            df["source"] = df["query"].apply(lambda q: "Google AI Overview" if any(q.lower() in s.lower() or s.lower() in q.lower() for s in seeds) else "Synthetic")
            def row_style(row):
                return ["background-color: lightblue" if row.source == "Google AI Overview" else "" for _ in row]
            st.subheader("🧠 Final Queries")
            st.dataframe(df.style.apply(row_style, axis=1))
        else:
            st.warning("No queries generated.")
