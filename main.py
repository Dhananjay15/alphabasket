from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import json
import re
from cohere import Client

app = FastAPI()

# === Setup Cohere and Supabase ===
co = Client(os.getenv("COHERE_API_KEY"))
SUPABASE_URL = "https://pxwbanyqpfhwwngqinfr.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# === Request Schema ===
class PromptInput(BaseModel):
    prompt: str

# === Helpers ===
def parse_numeric_filter(expr: str, default_value: float, default_op: str = "gte") -> tuple[str, float]:
    if not isinstance(expr, str):
        return default_op, default_value

    match = re.match(r'([<>]=?|=)?\s*([0-9.]+)', expr.strip())
    if match:
        raw_op = match.group(1) or default_op
        value = float(match.group(2))

        op_map = {
            '>': 'gt',
            '>=': 'gte',
            '<': 'lt',
            '<=': 'lte',
            '=': 'eq',
        }
        op = op_map.get(raw_op, default_op)
        return op, value

    return default_op, default_value

# === PromptParserAgent ===
def parse_prompt(prompt: str) -> dict:
    response = co.chat(
        message=f"""
You are an elite financial reasoning agent trained to parse unstructured investor prompts into structured, machine-readable screening criteria. 
You understand investment styles, fundamental metrics, macroeconomic drivers, sector trends, ESG preferences, and modern portfolio theory.

Parse the prompt below and return JSON:
\"\"\"{prompt}\"\"\"

Only return valid JSON in this format:
{{
  "theme": "...",
  "keywords": ["..."],
  "horizon": "short-term | mid-term | long-term",
  "filters": {{
    "sector": "...",
    "industry": "...",
    "roe": "...", 
    "pe_ratio": "...",
    "debt_equity": "...",
    "market_cap": "...",
    "dividend_yield": "..."
  }}
}}""",
        temperature=0.3
    )
    match = re.search(r'{[\s\S]*}', response.text.strip())
    if not match:
        raise ValueError("Cohere response did not contain valid JSON block")
    return json.loads(match.group(0))

# === StockFilterAgent ===
def filter_stocks(filters: dict):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    roe_op, roe_val = parse_numeric_filter(filters.get("roe"), 0)
    debt_op, debt_val = parse_numeric_filter(filters.get("debt_equity"), 1)
    mc_op, mc_val = parse_numeric_filter(filters.get("market_cap"), 1e12)

    params = {
        f"roe": f"{roe_op}.{roe_val}",
        f"debt_equity": f"{debt_op}.{debt_val}",
        f"market_cap": f"{mc_op}.{mc_val}",
    }

    sector = filters.get("sector")
    if isinstance(sector, str) and sector.strip():
        params["sector"] = f"ilike.%{sector.strip()}%"

    print("🧾 Supabase filters:", params)
    response = requests.get(f"{SUPABASE_URL}/rest/v1/nse500_stocks", headers=headers, params=params)

    try:
        data = response.json()
        if isinstance(data, dict) and "message" in data:
            raise ValueError(f"Supabase error: {data['message']}")
        if not isinstance(data, list):
            raise ValueError("Expected list of stock records")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse Supabase response: {e}")

# === StockScorerAgent ===
def score_stock(stock, filters):
    score = 0
    if stock.get('roe') and float(stock['roe']) >= float(filters.get('roe', '15').strip('><=')):
        score += 20
    if stock.get('debt_equity') and float(stock['debt_equity']) <= float(filters.get('debt_equity', '0.5').strip('><=')):
        score += 20
    if stock.get('pe_ratio') and float(stock['pe_ratio']) < 30:
        score += 15
    if filters.get("theme", "").lower() in stock.get("industry", "").lower():
        score += 20
    return score

# === ReasoningAgent ===
def generate_reasoning(stock, prompt):
    summary = f"{stock['name']} ({stock['symbol']}) in {stock['industry']}"
    message = f"User Prompt: {prompt}\nWhy was this stock selected: {summary}? Provide a short financial explanation."
    res = co.chat(message=message, temperature=0.5)
    return res.text.strip()

# === BasketBuilderAgent ===
def build_basket(stocks, top_n=10):
    return sorted(stocks, key=lambda x: x['score'], reverse=True)[:top_n]

# === API Endpoint ===
@app.post("/generate_basket")
async def generate_basket(input: PromptInput):
    try:
        parsed = parse_prompt(input.prompt)
    except Exception as e:
        return {"error": f"PromptParserAgent failed: {e}"}

    try:
        filtered = filter_stocks(parsed['filters'])
    except Exception as e:
        return {"error": f"StockFilterAgent failed: {e}"}

    enriched = []
    for s in filtered:
        if not isinstance(s, dict):
            print("⚠️ Skipping non-dict stock:", s)
            continue
        try:
            s['score'] = score_stock(s, parsed['filters'])
            s['reason'] = generate_reasoning(s, input.prompt)
            enriched.append(s)
        except Exception as e:
            print(f"⚠️ Failed to process stock {s.get('symbol', '?')}: {e}")

    try:
        top_stocks = build_basket(enriched)
    except Exception as e:
        return {"error": f"BasketBuilderAgent failed: {e}"}

    return {
        "basket": top_stocks,
        "summary": {
            "theme": parsed.get("theme"),
            "horizon": parsed.get("horizon"),
            "filters": parsed.get("filters")
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}
