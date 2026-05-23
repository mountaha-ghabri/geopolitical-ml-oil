import requests
import time
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import sys
import os

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
API_KEY = "CjAWFnLjvzXpxRjnv0BbqjpnRLnI2N2qsPAwZbT0wA4AwGCO"
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
FINAL_START = datetime(1990, 1, 1)   # naive – no timezone
END_DATE    = datetime(2022, 12, 31) # naive

QUERY = '("United States" OR "US" OR "U.S.") AND ("China" OR "Chinese")'

TEMP_CSV = "nyt_us_china_articles_temp.csv"

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def wait_until_next_utc_day():
    now_utc = datetime.now(timezone.utc)
    next_day = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    seconds = (next_day - now_utc).total_seconds() + 10
    print(f"⏳ Daily quota exhausted. Sleeping {seconds/3600:.1f} hours until next UTC day...")
    time.sleep(seconds)


def fetch_page(url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                json_body = resp.json()
                fault = json_body.get("fault", {}).get("faultstring", "").lower()
                if "quota limit" in fault or "daily" in fault:
                    wait_until_next_utc_day()
                    continue
                else:
                    print("  ⏳ Per-minute rate limit – sleeping 60s...")
                    time.sleep(60)
                    continue
            elif resp.status_code in (401, 403):
                print("  🔒 Authentication error – check your API key.")
                return None
            else:
                print(f"  ⚠ HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"  ⚠ Network error: {e}")
            time.sleep(5)
    return None


def parse_articles(data: dict) -> List[Dict]:
    resp = data.get("response", {})
    docs = resp.get("docs") or []
    parsed = []
    for doc in docs:
        parsed.append({
            "pub_date": doc.get("pub_date", ""),
            "headline": doc.get("headline", {}).get("main", ""),
            "lead_paragraph": doc.get("lead_paragraph", ""),
            "abstract": doc.get("abstract", ""),
            "section_name": doc.get("section_name", ""),
            "desk": doc.get("desk", ""),
            "type_of_material": doc.get("type_of_material", ""),
            "web_url": doc.get("web_url", ""),
        })
    return parsed


def load_progress() -> (List[Dict], datetime):
    """Load existing temp CSV; return articles list and next start month (naive)."""
    articles = []
    next_start = FINAL_START
    if os.path.exists(TEMP_CSV):
        try:
            df = pd.read_csv(TEMP_CSV)
            if not df.empty and "pub_date" in df.columns:
                df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
                df = df.dropna(subset=["pub_date"]).sort_values("pub_date")
                if not df.empty:
                    # Make sure pub_date is naive (remove timezone if present)
                    if df["pub_date"].dt.tz is not None:
                        df["pub_date"] = df["pub_date"].dt.tz_localize(None)
                    articles = df.to_dict(orient="records")
                    last_date = df["pub_date"].max()
                    print(f"📂 Loaded {len(articles)} articles from {TEMP_CSV}")
                    print(f"   Last collected date: {last_date.date()}")
                    # Next start: first day of month after last date
                    next_start = (last_date.replace(day=1) + timedelta(days=32)).replace(day=1)
                    if next_start > END_DATE:
                        print("✅ All months already collected!")
                        return articles, next_start
        except Exception as e:
            print(f"⚠ Could not read checkpoint file: {e}")
            articles = []
    return articles, next_start


# -----------------------------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------------------------
print("🚀 Starting NYT US‑China article collector (with resume)")
print(f"   Target range: {FINAL_START.date()} to {END_DATE.date()}")

# 1. Quick API test
print("\n🧪 Testing API with 2022-01 ...")
test_params = {
    "q": QUERY,
    "begin_date": "20220101",
    "end_date": "20220131",
    "page": 0,
    "sort": "oldest",
    "api-key": API_KEY,
}
data = fetch_page(BASE_URL, test_params)
if data is None:
    print("❌ API test failed. Check key or network. Exiting.")
    sys.exit(1)
resp = data.get("response", {})
print(f"✅ API OK – keys: {list(resp.keys())}")
articles_test = parse_articles(data)
if articles_test:
    print(f"📰 Sample headline: {articles_test[0]['headline']}")

# 2. Load checkpoint
all_articles, current_start = load_progress()
if current_start > END_DATE:
    print("All months already collected. Run the final save script if needed.")
    sys.exit(0)

print(f"\n⏩ Resuming from {current_start.strftime('%Y-%m')} ...")

# 3. Main collection loop
while current_start < END_DATE:
    next_month = current_start.replace(day=28) + timedelta(days=4)
    current_end = min(next_month - timedelta(days=next_month.day), END_DATE)

    str_start = current_start.strftime("%Y%m%d")
    str_end   = current_end.strftime("%Y%m%d")

    print(f"\n📅 Collecting {current_start.strftime('%Y-%m')} ...")
    month_articles = 0

    for page in range(0, 100):
        params = {
            "q": QUERY,
            "begin_date": str_start,
            "end_date": str_end,
            "page": page,
            "sort": "oldest",
            "api-key": API_KEY,
        }
        data = fetch_page(BASE_URL, params)
        if data is None:
            break

        articles_page = parse_articles(data)
        if not articles_page:
            break

        all_articles.extend(articles_page)
        month_articles += len(articles_page)
        print(f"  Page {page}: {len(articles_page)} articles (month total: {month_articles})")
        time.sleep(12)

    if month_articles > 0:
        pd.DataFrame(all_articles).to_csv(TEMP_CSV, index=False)
        print(f"💾 Saved {len(all_articles)} total articles so far.")
    else:
        print(f"ℹ️ No articles found for {current_start.strftime('%Y-%m')}.")

    current_start = next_month

# 4. Final save
df = pd.DataFrame(all_articles)
if not df.empty:
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    # Ensure naive for consistency
    if df["pub_date"].dt.tz is not None:
        df["pub_date"] = df["pub_date"].dt.tz_localize(None)
    df = df.sort_values("pub_date").reset_index(drop=True)

print(f"\n🎉 Done! Total articles: {len(df)}")
print(f"📆 Date range: {df['pub_date'].min()} to {df['pub_date'].max()}")
df.to_csv("nyt_us_china_headlines.csv", index=False, encoding="utf-8")
print("💾 Final CSV saved to nyt_us_china_headlines.csv")