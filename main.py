# main.py
import os
import re
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx
import cohere
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

# ---------- CONFIG ----------

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://pxwbanyqpfhwwngqinfr.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # must be set server-side
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_KEY env var required")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY env var required")

# Supabase rest endpoint (table)
STOCK_TABLE = "nse500_stocks"

# Limits & batch sizes
SUPABASE_FETCH_LIMIT = int(os.getenv("SUPABASE_FETCH_LIMIT", "300"))
REASON_BATCH_SIZE = int(os.getenv("REASON_BATCH_SIZE", "10"))
REASON_CACHE_TTL_SECONDS = int(os.getenv("REASON_CACHE_TTL_SECONDS", "3600"))

# Scoring config (tunable)
SCORING_WEIGHTS = {
    "roe": 0.35,          # relative contribution
    "pe": 0.25,
    "debt_equity": 0.2,
    "sector_match": 0.1,
    "industry_match": 0.1
}
# Targets / caps used in normalization
SCORING_TARGETS = {
    "roe_cap": 30.0,      # ROE above this considered max contribution
    "pe_cap": 50.0,
    "debt_cap": 2.0
}

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("basket_gen")

# ---------- FASTAPI ----------
app = FastAPI(title="AlphaBasket Basket Generator")

# ---------- Cohere Client (blocking) ----------
co = cohere.Client(COHERE_API_KEY)

# ---------- Simple in-memory TTL cache for reasoning ----------
class TTLCache:
    def __init__(self):
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str):
        async with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expiry_ts, value = item
            if expiry_ts < asyncio.get_event_loop().time():
                del self._store[key]
                return None
            return value

    async def set(self, key: str, value: Any, ttl: int = REASON_CACHE_TTL_SECONDS):
        async with self._lock:
            self._store[key] = (asyncio.get_event_loop().time() + ttl, value)

reason_cache = TTLCache()

# ---------- Pydantic types for prompt parser output ----------
class ParsedFilters(BaseModel):
    sector: Optional[str] = ""
    industry: Optional[str] = ""
    roe: Optional[str] = ""
    pe_ratio: Optional[str] = ""
    debt_equity: Optional[str] = ""
    market_cap: Optional[str] = ""
    dividend_yield: Optional[str] = ""

class ParsedPrompt(BaseModel):
    theme: Optional[str] = ""
    keywords: Optional[List[str]] = []
    horizon: Optional[str] = "mid-term"
    filters: ParsedFilters = ParsedFilters()

# ---------- API request model ----------
class PromptInput(BaseModel):
    prompt: str = Field(..., min_length=3)

# ---------- helpers ----------
OP_MAP = {
    '>': 'gt',
    '>=': 'gte',
    '<': 'lt',
    '<=': 'lte',
    '=': 'eq'
}

def parse_numeric_filter(expr: Optional[str], default_op: str = "gte", default_val: Optional[float] = None) -> Tuple[str, Optional[float]]:
    """
    Convert strings like '>=15' or '15' -> ('gte', 15.0)
    If expr is None or empty, returns (default_op, default_val)
    """
    if not expr and default_val is not None:
        return default_op, float(default_val)
    if not expr:
        return default_op, None
    if isinstance(expr, (int, float)):
        return default_op, float(expr)
    s = str(expr).strip()
    m = re.match(r'^([<>]=?|=)?\s*([0-9]+(?:\.[0-9]+)?)$', s)
    if not m:
        return default_op, None
    raw_op = m.group(1) or default_op
    num = float(m.group(2))
    op = OP_MAP.get(raw_op, default_op)
    return op, num

def normalize_score_components(stock: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert raw fields to normalized scores between 0 and 1 for each metric
    """
    try:
        roe = float(stock.get("roe") or 0)
    except Exception:
        roe = 0.0
    try:
        pe = float(stock.get("pe_ratio") or SCORING_TARGETS["pe_cap"])
    except Exception:
        pe = SCORING_TARGETS["pe_cap"]
    try:
        de = float(stock.get("debt_equity") or SCORING_TARGETS["debt_cap"])
    except Exception:
        de = SCORING_TARGETS["debt_cap"]

    roe_score = max(0.0, min(roe / SCORING_TARGETS["roe_cap"], 1.0))
    # For PE: lower is better. invert and cap
    pe_score = max(0.0, min((SCORING_TARGETS["pe_cap"] - pe) / SCORING_TARGETS["pe_cap"], 1.0))
    # For debt: lower is better. If debt is 0 => score 1, if debt >= cap => 0
    de_score = max(0.0, min((SCORING_TARGETS["debt_cap"] - de) / SCORING_TARGETS["debt_cap"], 1.0))

    return {"roe_score": roe_score, "pe_score": pe_score, "de_score": de_score}

def explain_score(stock: Dict[str, Any], components: Dict[str, float], final_score: float) -> str:
    # Simple human readable explanation
    return (
        f"ROE {stock.get('roe', 'N/A')} (score {components['roe_score']:.2f}), "
        f"PE {stock.get('pe_ratio', 'N/A')} (score {components['pe_score']:.2f}), "
        f"Debt/Equity {stock.get('debt_equity', 'N/A')} (score {components['de_score']:.2f}). "
        f"Final score {final_score:.2f}."
    )

async def run_cohere_chat(prompt_text: str, temperature: float = 0.3, max_retries: int = 2) -> str:
    """
    Cohere's client is blocking. Run it in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()

    def _call():
        resp = co.chat(message=prompt_text, temperature=temperature)
        return resp.text

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await loop.run_in_executor(None, _call)
        except Exception as e:
            last_exc = e
            logger.warning("Cohere call failed, attempt %d: %s", attempt + 1, e)
            await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Cohere chat failed after {max_retries + 1} attempts: {last_exc}")

# ---------- Core pipeline steps ----------

async def parse_prompt_agent(prompt: str) -> ParsedPrompt:
    """
    Use Cohere to convert NL prompt -> structured JSON (validated by Pydantic).
    Includes retry and basic sanitize.
    """
    system_message = f"""
You are a financial reasoning agent. Convert the user's investment prompt into a strict JSON object.
Return only a JSON object that follows the schema:

{{
  "theme": string,
  "keywords": [string],
  "horizon": "short-term" | "mid-term" | "long-term",
  "filters": {{
     "sector": string,
     "industry": string,
     "roe": string,
     "pe_ratio": string,
     "debt_equity": string,
     "market_cap": string,
     "dividend_yield": string
  }}
}}

Rules:
- Use numeric filters only (e.g. 'roe': '>=15', 'pe_ratio': '<=25').
- If the prompt uses vague terms (like mid-cap, high ROE), convert to numeric ranges reasonably.
- Keep values safe for JSON parsing. No comments or text outside JSON.
Prompt: {json.dumps(prompt)}
"""
    raw = await run_cohere_chat(system_message, temperature=0.25)
    # try to extract first JSON block
    js_text = None
    m = re.search(r'(\{[\s\S]*\})', raw.strip())
    if m:
        js_text = m.group(1)
    else:
        # fallback: if raw looks like json start to end try it
        if raw.strip().startswith("{"):
            js_text = raw.strip()

    if not js_text:
        logger.error("Cohere returned non-JSON: %s", raw)
        raise HTTPException(status_code=502, detail="Failed to parse prompt into JSON")

    try:
        parsed_dict = json.loads(js_text)
        parsed = ParsedPrompt(**parsed_dict)
        return parsed
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error("Invalid JSON from LLM: %s\nraw: %s", e, raw)
        raise HTTPException(status_code=502, detail=f"Invalid JSON from LLM: {e}")

async def query_supabase(filters: ParsedFilters, limit: int = SUPABASE_FETCH_LIMIT) -> List[Dict]:
    """
    Build PostgREST params dynamically and fetch rows async from Supabase REST endpoint.
    """
    params: Dict[str, Any] = {"select": "*", "limit": str(limit)}
    # numeric filters
    roe_op, roe_val = parse_numeric_filter(filters.roe)
    if roe_val is not None:
        params["roe"] = f"{roe_op}.{roe_val}"

    pe_op, pe_val = parse_numeric_filter(filters.pe_ratio)
    if pe_val is not None:
        params["pe_ratio"] = f"{pe_op}.{pe_val}"

    de_op, de_val = parse_numeric_filter(filters.debt_equity)
    if de_val is not None:
        params["debt_equity"] = f"{de_op}.{de_val}"

    mc_op, mc_val = parse_numeric_filter(filters.market_cap)
    if mc_val is not None:
        # supabase expects bigint for market cap; ensure integer
        params["market_cap"] = f"{mc_op}.{int(mc_val)}"

    # string filters (case insensitive)
    if filters.sector and filters.sector.strip():
        params["sector"] = f"ilike.%{filters.sector.strip()}%"
    if filters.industry and filters.industry.strip():
        params["industry"] = f"ilike.%{filters.industry.strip()}%"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    url = f"{SUPABASE_URL}/rest/v1/{STOCK_TABLE}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, params=params, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("Supabase error: %s %s", r.status_code, r.text)
            raise HTTPException(status_code=502, detail=f"Supabase error: {r.text[:400]}")
        data = r.json()
        if isinstance(data, dict) and "message" in data:
            logger.error("Supabase returned error payload: %s", data)
            raise HTTPException(status_code=502, detail=f"Supabase error: {data.get('message')}")
        if not isinstance(data, list):
            logger.error("Unexpected supabase response type: %s", type(data))
            raise HTTPException(status_code=502, detail="Unexpected Supabase response")
        return data

def score_stock(stock: Dict[str, Any], parsed_filters: ParsedFilters) -> Dict[str, Any]:
    """
    Return a dict with detailed score components and final normalized score (0-100).
    """
    comps = normalize_score_components(stock)
    # base weighted score
    score = (
        comps["roe_score"] * SCORING_WEIGHTS["roe"] +
        comps["pe_score"] * SCORING_WEIGHTS["pe"] +
        comps["de_score"] * SCORING_WEIGHTS["debt_equity"]
    )
    # sector / industry matches (binary)
    sector = parsed_filters.sector or ""
    industry = parsed_filters.industry or ""
    sector_match = 1.0 if sector and sector.lower() in (stock.get("sector") or "").lower() else 0.0
    industry_match = 1.0 if industry and industry.lower() in (stock.get("industry") or "").lower() else 0.0

    score += sector_match * SCORING_WEIGHTS["sector_match"]
    score += industry_match * SCORING_WEIGHTS["industry_match"]

    # normalize to 0-100
    final_score = max(0.0, min(score, 1.0)) * 100.0
    details = {
        "components": {
            "roe_score": comps["roe_score"],
            "pe_score": comps["pe_score"],
            "de_score": comps["de_score"],
            "sector_match": sector_match,
            "industry_match": industry_match
        },
        "final_score": round(final_score, 2),
        "explain": explain_score(stock, comps, final_score)
    }
    return details

async def generate_reasons_batch(stocks: List[Dict], prompt_text: str) -> Dict[str, str]:
    """
    Generate reason strings for a list of stock dicts (batched).
    Uses caching to avoid repeated LLM calls for same symbol+prompt combo.
    """
    results: Dict[str, str] = {}

    # check cache first
    to_query = []
    for s in stocks:
        key = f"reason:{s.get('symbol')}:{hash(prompt_text)}"
        cached = await reason_cache.get(key)
        if cached:
            results[s.get("symbol")] = cached
        else:
            to_query.append((s, key))

    if not to_query:
        return results

    # Prepare single prompt for batch reasoning to reduce cost:
    # we'll ask the model to produce a JSON map {symbol: reason}
    items_text = "\n".join([f"{i+1}. {s.get('symbol')} - {s.get('name')} (ROE: {s.get('roe', 'N/A')}, PE: {s.get('pe_ratio','N/A')})"
                            for i, (s, _) in enumerate(to_query)])
    sys_prompt = (
        f"User Prompt: {prompt_text}\n\n"
        "For each stock listed below, produce a concise 1-2 line reason why it fits the user's prompt. "
        "Mention ROE, PE or valuation strength if relevant and a main risk. "
        "Return a strict JSON object mapping symbols to plain text reasons.\n\n"
        f"Stocks:\n{items_text}\n\n"
        "Example output:\n{ \"AAPL\": \"Reason...\", \"MSFT\": \"Reason...\" }\n"
    )
    raw = await run_cohere_chat(sys_prompt, temperature=0.45)

    # extract JSON from response
    m = re.search(r'(\{[\s\S]*\})', raw)
    if not m:
        logger.warning("Batch reasoner returned non-JSON, falling back to per-stock templating")
        # fallback: do simple templated reasons for each
        for s, key in to_query:
            reason = f"{s.get('name')} ({s.get('symbol')}): ROE {s.get('roe','N/A')}, PE {s.get('pe_ratio','N/A')}. Risk: sector volatility."
            results[s.get("symbol")] = reason
            await reason_cache.set(key, reason)
        return results

    try:
        mapping = json.loads(m.group(1))
        for s, key in to_query:
            reason = mapping.get(s.get("symbol")) or mapping.get(s.get("symbol").upper()) or ""
            if not reason:
                reason = f"{s.get('name')} ({s.get('symbol')}): ROE {s.get('roe','N/A')}, PE {s.get('pe_ratio','N/A')}. Risk: sector volatility."
            results[s.get("symbol")] = reason
            await reason_cache.set(key, reason)
        return results
    except Exception as e:
        logger.exception("Failed to parse batch reasons: %s", e)
        # fallback templated reasons
        for s, key in to_query:
            reason = f"{s.get('name')} ({s.get('symbol')}): ROE {s.get('roe','N/A')}, PE {s.get('pe_ratio','N/A')}. Risk: sector volatility."
            results[s.get("symbol")] = reason
            await reason_cache.set(key, reason)
        return results

# ---------- Endpoint ----------

@app.post("/generate_basket")
async def generate_basket(input: PromptInput):
    # 1) Parse prompt -> filters
    try:
        parsed = await parse_prompt_agent(input.prompt)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prompt parsing failed: %s", e)
        raise HTTPException(status_code=502, detail="Prompt parsing failed")

    # 2) Query supabase
    try:
        rows = await query_supabase(parsed.filters)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Supabase query failed: %s", e)
        raise HTTPException(status_code=502, detail="Stock fetch failed")

    if not rows:
        return {"basket": [], "summary": {"theme": parsed.theme, "horizon": parsed.horizon, "filters": parsed.filters.dict()}}

    # 3) Score stocks locally
    enriched = []
    for r in rows:
        details = score_stock(r, parsed.filters)
        r["_score"] = details["final_score"]
        r["_score_details"] = details
        enriched.append(r)

    # 4) Build preliminary sorted list and cut to top_k for reasoning
    enriched_sorted = sorted(enriched, key=lambda x: x["_score"], reverse=True)
    top_for_reason = enriched_sorted[:REASON_BATCH_SIZE]

    # 5) Batch generate reasons for top results
    try:
        prompt_text = input.prompt
        reasons_map = await generate_reasons_batch(top_for_reason, prompt_text)
    except Exception as e:
        logger.exception("Reason generation failed: %s", e)
        reasons_map = {}

    # attach reasons to top stocks
    for s in enriched_sorted:
        sym = s.get("symbol")
        if sym in reasons_map:
            s["reason"] = reasons_map[sym]
        else:
            # if not in top or missing, keep a short templated reason
            s["reason"] = s.get("_score_details", {}).get("explain", "")

    # 6) Build final basket (top N)
    TOP_N = 10
    basket = enriched_sorted[:TOP_N]

    # 7) Compute summary metrics
    def safe_float(x): 
        try: return float(x or 0)
        except: return 0.0

    avg_roe = round(sum(safe_float(s.get("roe")) for s in basket) / len(basket), 2) if basket else 0
    avg_pe = round(sum(safe_float(s.get("pe_ratio")) for s in basket) / len(basket), 2) if basket else 0

    # return explainable output
    return {
        "basket": [
            {
                "symbol": s.get("symbol"),
                "name": s.get("name"),
                "sector": s.get("sector"),
                "industry": s.get("industry"),
                "price": s.get("price"),
                "market_cap": s.get("market_cap"),
                "pe_ratio": s.get("pe_ratio"),
                "roe": s.get("roe"),
                "debt_equity": s.get("debt_equity"),
                "score": s.get("_score"),
                "score_breakdown": s.get("_score_details", {}),
                "reason": s.get("reason"),
                "raw": s  # include raw for debugging / optional
            } for s in basket
        ],
        "summary": {
            "theme": parsed.theme,
            "horizon": parsed.horizon,
            "filters": parsed.filters.dict(),
            "average_roe": avg_roe,
            "average_pe": avg_pe,
            "returned_count": len(basket)
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
