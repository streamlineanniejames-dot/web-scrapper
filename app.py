import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import time

st.set_page_config(page_title="Bulk Website AI Insights – Batch Version", layout="wide")

GROQ_KEY = st.secrets.get("GROQ_API_KEY", None)
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

if not GROQ_KEY:
    st.error("❌ Missing GROQ_API_KEY in Streamlit Secrets.")

# --------------------
# Scraper with fallback
# --------------------
def try_fetch(url):
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            return r.text
    except:
        return None
    return None


def scrape_site(url):
    raw = url.strip()
    if raw.startswith("http"):
        attempts = [raw]
    else:
        base = raw.replace("www.", "")
        attempts = [f"https://{base}", f"http://{base}", f"https://www.{base}", f"http://www.{base}"]

    for link in attempts:
        html = try_fetch(link)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)
            return text[:4000], link
    return "SCRAPE_ERROR: Unable to fetch site", None

# --------------------
# AI Insights
# --------------------
def get_ai_insights(url, scraped_text):
    prompt = f"""
Extract B2B-focused insights only.
Avoid B2C except HNWI / UHNWI.
Return ONLY VALID JSON.

JSON format:
{{
"company_name": "",
"company_summary": "",
"main_products": [],
"ideal_customers": [],
"ideal_audience": [],
"industry": "",
"countries_of_operation": []
}}

Website: {url}
Content: {scraped_text}
"""

    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}

    try:
        r = requests.post(API_URL, json=body, headers=headers, timeout=30)
        resp = r.json()
        if "choices" not in resp:
            return {"error": resp}
        raw = resp["choices"][0]["message"]["content"]

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == -1:
            return {"error": "Invalid AI JSON"}
        return json.loads(raw[start:end])
    except Exception as e:
        return {"error": str(e)}

# --------------------
# Processor (batch 50)
# --------------------
def process_csv(df, website_column, live_box, batch_size=50):
    if 'status' not in df.columns:
        df['status'] = ''

    results = []

    to_process = df[df['status'] != 'done'].head(batch_size)

    for idx, row in to_process.iterrows():
        site = str(row[website_column]).strip()
        live_box.markdown(f"## 🔍 Processing {idx+1}/{len(df)} – `{site}`")

        scraped_text, final_url = scrape_site(site)
        live_box.write(f"🌐 Using URL: {final_url or 'Not Found'}")
        live_box.write(scraped_text[:500] + "...")

        ai_data = get_ai_insights(final_url or site, scraped_text)
        live_box.json(ai_data)

        for k, v in ai_data.items():
            row[k] = v
        row['status'] = 'done'
        results.append(row)

        time.sleep(30)

    final_df = pd.concat([df[df['status']=='done'], pd.DataFrame(results)]).drop_duplicates()
    return final_df

# --------------------
# UI
# --------------------
st.title("🌍 Bulk Website → AI Insights (Batch 50 per run)")
file = st.file_uploader("📤 Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.write("### Preview")
    st.dataframe(df.head())

    auto_col = next((c for c in df.columns if c.lower() in ["website", "url", "domain"]), df.columns[0])
    website_column = st.selectbox("Website Column", df.columns, index=list(df.columns).index(auto_col))

    live_box = st.empty()

    if st.button("🚀 Start Processing Batch", use_container_width=True):
        with st.spinner("Processing batch..."):
            final_df = process_csv(df, website_column, live_box, batch_size=50)
        st.success("🎉 Batch Completed!")
        st.dataframe(final_df)
        st.download_button("📥 Download CSV", data=final_df.to_csv(index=False).encode('utf-8'), file_name="ai_batch_insights.csv", mime="text/csv")
