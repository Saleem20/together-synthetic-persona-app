from __future__ import annotations

import json
import os
import time
from statistics import mean

import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Together Synthetic Persona Lab", layout="wide")

PERSONAS = [
    {"id": "mum", "name": "Sarah M.", "label": "Health-First Mum", "country": "AU", "population_share": 0.30},
    {"id": "senior", "name": "Raj P.", "label": "Budget Senior", "country": "AU", "population_share": 0.25},
    {"id": "exec", "name": "Marcus B.", "label": "Busy Executive", "country": "US", "population_share": 0.20},
    {"id": "genz", "name": "Aisha K.", "label": "Gen Z Ingredient Hunter", "country": "UK", "population_share": 0.15},
    {"id": "prag", "name": "David L.", "label": "Skeptical Pragmatist", "country": "US", "population_share": 0.10},
]

SUBCATEGORIES = {
    "Oral Health": ["Daily toothpaste", "Whitening", "Sensitivity", "Mouthwash / rinse"],
    "OTC": ["Pain relief / analgesics", "Cold & flu", "Digestive", "Allergy / hay fever"],
    "Wellness": ["Adult multivitamins", "Immunity", "Sleep & mood", "Sports nutrition"],
}

# Practical budget defaults; user can override in sidebar.
DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def get_together_api_key() -> str:
    try:
        key = st.secrets.get("TOGETHER_API_KEY", "")  # type: ignore[attr-defined]
        if key:
            return key
    except Exception:
        pass
    return os.getenv("TOGETHER_API_KEY", "")


def build_messages(persona: dict, category: str, subcategory: str, subject: str) -> list[dict]:
    system = (
        f"You are {persona['name']} ({persona['label']}) from {persona['country']}. "
        "Respond like a real consumer in first person. Output valid JSON only."
    )
    user = f"""
Evaluate this claim/name for {category} -> {subcategory}.

Subject: "{subject}"

Return strictly this JSON schema:
{{
  "scores": {{
    "believability": 1-7,
    "relevance": 1-7,
    "clarity": 1-7,
    "differentiation": 1-7,
    "purchase_intent": 1-7
  }},
  "verbatim": "2 short sentences",
  "top_positive": "one sentence",
  "top_concern": "one sentence"
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("No JSON found in model output.")
        return json.loads(cleaned[start : end + 1])


def evaluate_persona(client: OpenAI, model_name: str, persona: dict, category: str, subcategory: str, subject: str) -> dict:
    messages = build_messages(persona, category, subcategory, subject)
    last_error = ""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.4,
                max_tokens=400,
            )
            content = response.choices[0].message.content or ""
            payload = extract_json(content)
            scores = payload["scores"]
            overall = round(
                mean(
                    [
                        float(scores["believability"]),
                        float(scores["relevance"]),
                        float(scores["clarity"]),
                        float(scores["differentiation"]),
                        float(scores["purchase_intent"]),
                    ]
                ),
                2,
            )
            return {
                "ok": True,
                "error": "",
                "persona": persona["name"],
                "segment": persona["label"],
                "country": persona["country"],
                "population_share": persona["population_share"],
                "overall": overall,
                "believability": int(round(float(scores["believability"]))),
                "relevance": int(round(float(scores["relevance"]))),
                "clarity": int(round(float(scores["clarity"]))),
                "differentiation": int(round(float(scores["differentiation"]))),
                "purchase_intent": int(round(float(scores["purchase_intent"]))),
                "verbatim": payload.get("verbatim", ""),
                "top_positive": payload.get("top_positive", ""),
                "top_concern": payload.get("top_concern", ""),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1))

    return {
        "ok": False,
        "error": last_error or "Unknown error",
        "persona": persona["name"],
        "segment": persona["label"],
        "country": persona["country"],
        "population_share": persona["population_share"],
    }


def run_batch(client: OpenAI, model_name: str, personas: list[dict], category: str, subcategory: str, subject: str) -> list[dict]:
    rows = []
    # Deliberately sequential for reliability on free/low-rate keys.
    for persona in personas:
        rows.append(evaluate_persona(client, model_name, persona, category, subcategory, subject))
        time.sleep(0.3)
    return rows


def compute_weighted_mean(df: pd.DataFrame) -> float:
    weighted_sum = (df["overall"] * df["population_share"]).sum()
    total_share = df["population_share"].sum()
    if total_share <= 0:
        return round(df["overall"].mean(), 2)
    return round(weighted_sum / total_share, 2)


def main() -> None:
    st.title("Together Synthetic Persona Lab")
    st.caption("New robust app using Together API with backend-managed key and budget-first defaults.")

    api_key = get_together_api_key()
    if not api_key:
        st.error("Missing TOGETHER_API_KEY in Streamlit secrets or environment.")
        st.stop()

    client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")

    st.sidebar.header("Setup")
    model_name = st.sidebar.selectbox("Model (cheap defaults)", DEFAULT_MODELS, index=0)
    countries = st.sidebar.multiselect("Country filter", ["AU", "US", "UK"], default=["AU", "US"])
    max_personas = st.sidebar.slider("Max personas per run", min_value=1, max_value=5, value=2, step=1)
    selected_personas = [p for p in PERSONAS if p["country"] in countries][:max_personas]

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category", list(SUBCATEGORIES.keys()))
    with col2:
        subcategory = st.selectbox("Sub-category", SUBCATEGORIES[category])

    subject = st.text_area("Claim or name", placeholder="Type claim or product name...")
    run = st.button("Run test", type="primary", use_container_width=True)

    if not run:
        return
    if not subject.strip():
        st.warning("Enter a claim or name first.")
        st.stop()
    if not selected_personas:
        st.warning("Select at least one persona.")
        st.stop()

    with st.spinner("Running persona evaluations..."):
        rows = run_batch(client, model_name, selected_personas, category, subcategory, subject.strip())

    ok_rows = [row for row in rows if row.get("ok")]
    err_rows = [row for row in rows if not row.get("ok")]

    if err_rows:
        st.warning(f"{len(err_rows)} persona call(s) failed. Showing successful results only.")
        st.dataframe(pd.DataFrame(err_rows)[["persona", "country", "error"]], use_container_width=True)

    if not ok_rows:
        st.error("All persona calls failed. Try a different model or retry after 1-2 minutes.")
        st.stop()

    df = pd.DataFrame(ok_rows)
    simple_mean = round(df["overall"].mean(), 2)
    weighted_mean = compute_weighted_mean(df)

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Simple overall", simple_mean)
    c2.metric("Population-weighted overall", weighted_mean)
    c3.metric("Successful personas", len(ok_rows))

    st.dataframe(
        df.drop(columns=["ok", "error", "verbatim", "top_positive", "top_concern"]),
        use_container_width=True,
    )

    st.subheader("Verbatims")
    for row in ok_rows:
        with st.expander(f"{row['persona']} — {row['segment']} ({row['country']})"):
            st.write(row["verbatim"])
            st.write(f"Top positive: {row['top_positive']}")
            st.write(f"Top concern: {row['top_concern']}")

    st.download_button(
        "Download run JSON",
        data=json.dumps(rows, indent=2).encode("utf-8"),
        file_name="together_persona_results.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
