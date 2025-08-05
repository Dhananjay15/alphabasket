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
def parse_prompt(prompt: str) -> dict:
    response = co.chat(
        message=f"""
You are an elite financial reasoning agent trained to parse unstructured investor prompts into structured, machine-readable screening criteria. 
You understand investment styles, fundamental metrics, macroeconomic drivers, sector trends, ESG preferences, and modern portfolio theory.

Your job is to analyze the user's investment idea and return a precise JSON object with:
- 🧠 Theme: one-line summary of the investment vision (e.g., “AI-powered large caps”, “Green energy turnaround bets”)
- 🔍 Keywords: investor-specified traits (e.g., “low debt”, “high ROE”, “mid-cap”, “ESG”)
- ⏳ Horizon: one of [short-term, mid-term, long-term] inferred from user context
- 📊 Filters: strict quantitative screeners across major fundamentals
  (You must only use numeric-compatible values. Avoid vague terms like "high", "low". Instead map them to conservative numeric thresholds.)

🎯 Example input:
"Looking for undervalued mid-cap tech stocks with strong ROE and low debt for long-term compounding."

🎯 Your response format (JSON):
{{
  "theme": "Mid-cap tech value compounding",
  "keywords": ["mid-cap", "low debt", "high ROE", "value investing"],
  "horizon": "long-term",
  "filters": {{
    "sector": "Technology",
    "roe": ">=18",
    "pe_ratio": "<=25",
    "debt_equity": "<=0.3",
    "market_cap": "<=50000",
    "dividend_yield": ">=0",
    "industry": ""
  }}
}}

💬 Now parse this prompt:
\"\"\"{prompt}\"\"\"

Respond ONLY with the JSON. No comments. No natural language explanations.
""",
        temperature=0.3
    )

    # Extract JSON block safely using regex
    match = re.search(r'\{[\s\S]*\}', response.text.strip())
    if not match:
        raise ValueError("No valid JSON block found in Cohere response.")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Cohere: {e}")



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
        "sector": f"ilike.%{filters.get('sector', '')}%",
    }

    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/nse500_stocks",
        headers=headers,
        params=params
    )

    print("🔍 Supabase status:", response.status_code)
    
    try:
        data = response.json()
        print("📦 Supabase returned:", data)

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
    # Step 1: Parse prompt
    try:
        parsed = parse_prompt(input.prompt)
    except Exception as e:
        return {"error": f"PromptParserAgent failed: {str(e)}"}

    # Step 2: Filter stocks from Supabase
    try:
        filtered = filter_stocks(parsed['filters'])
        if not isinstance(filtered, list):
            raise ValueError("Expected a list of stocks but got something else.")
    except Exception as e:
        return {"error": f"StockFilterAgent failed: {str(e)}"}

    # Step 3: Score + explain each stock
    enriched = []
    for s in filtered:
        if not isinstance(s, dict):
            print(f"⚠️ Skipping unexpected non-dict stock: {s}")
            continue

        try:
            s['score'] = score_stock(s, parsed['filters'])
            s['reason'] = generate_reasoning(s, input.prompt)
            enriched.append(s)
        except Exception as e:
            print(f"⚠️ Failed to score or explain stock {s.get('symbol', '?')}: {e}")
            continue

    # Step 4: Build final basket
    try:
        top_stocks = build_basket(enriched)
    except Exception as e:
        return {"error": f"BasketBuilderAgent failed: {str(e)}"}

    # Return full response
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
