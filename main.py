import json
import re
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from cohere import Client

app = FastAPI()

# Initialize Cohere client
co = Client(os.getenv("COHERE_API_KEY"))

# Supabase configuration
SUPABASE_URL = "https://pxwbanyqpfhwwngqinfr.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# === Request Schema ===
class PromptInput(BaseModel):
    prompt: str

# === PromptParserAgent ===
import json
import re

# === PromptParserAgent ===
def parse_prompt(prompt: str) -> dict:
    response = co.chat(
        message=f"""
You are a financial prompt parser. Your job is to extract the user's investment preferences from the prompt.

Prompt: "{prompt}"

Only return a valid JSON object with this structure:
{{
  "theme": "...",
  "keywords": ["...", "..."],
  "horizon": "...",
  "filters": {{
    "sector": "...",
    "industry": "...",
    "roe": "...",
    "pe_ratio": "...",
    "debt_equity": "...",
    "market_cap": "...",
    "dividend_yield": "..."
  }}
}}
""",
        temperature=0.3
    )

    # Extract the JSON block from response text
    match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
    if not match:
        raise ValueError("Could not extract JSON from Cohere response")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")


# === StockFilterAgent ===
def filter_stocks(filters: dict):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    params = {
        "roe": f"gte.{filters.get('roe', 0)}",
        "debt_equity": f"lte.{filters.get('debt_equity', 1)}",
        "market_cap": f"lte.{filters.get('market_cap', 999999999)}",
        "sector": f"ilike.%{filters.get('sector', '')}%"
    }
    res = requests.get(f"{SUPABASE_URL}/rest/v1/nse500_stocks", headers=headers, params=params)
    return res.json()

# === StockScorerAgent ===
def score_stock(stock, filters):
    score = 0
    if stock.get('roe') and float(stock['roe']) >= float(filters.get('roe', 15)):
        score += 20
    if stock.get('debt_equity') and float(stock['debt_equity']) <= float(filters.get('debt_equity', 0.5)):
        score += 20
    if stock.get('pe_ratio') and float(stock['pe_ratio']) < 30:
        score += 15
    if filters.get("theme", "").lower() in stock.get("industry", "").lower():
        score += 20
    return score

# === ReasoningAgent ===
def generate_reasoning(stock, prompt):
    stock_summary = f"{stock['name']} ({stock['symbol']}) in {stock['industry']}"
    msg = f"User Prompt: {prompt}\nWhy was this stock selected: {stock_summary}? Give a short, financial explanation."
    res = co.chat(message=msg, temperature=0.5)
    return res.text.strip()

# === BasketBuilderAgent ===
def build_basket(stocks, top_n=10):
    sorted_stocks = sorted(stocks, key=lambda x: x['score'], reverse=True)
    return sorted_stocks[:top_n]

# === API Endpoint ===
@app.post("/generate_basket")
async def generate_basket(input: PromptInput):
    parsed = parse_prompt(input.prompt)
    filtered = filter_stocks(parsed['filters'])

    enriched = []
    for s in filtered:
        s['score'] = score_stock(s, parsed['filters'])
        s['reason'] = generate_reasoning(s, input.prompt)
        enriched.append(s)

    top_stocks = build_basket(enriched)

    return {
        "basket": top_stocks,
        "summary": {
            "theme": parsed.get("theme"),
            "horizon": parsed.get("horizon"),
            "filters": parsed.get("filters")
        }
    }

# Optional: healthcheck endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
