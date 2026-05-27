# ===================================================================
# script to construct a monthly, newspaper-based GPR index for 11 dyads
# ===================================================================

import pandas as pd
import time
import random
from datetime import datetime, timedelta
from gdeltdoc import GdeltDoc, Filters  # pip install gdeltdoc

# -------------------------------
# 1. YOUR 11 DYADS WITH FIPS CODES
# -------------------------------
dyads = [
    # (country_a_code, country_b_code, dyad_name)
    ("CH", "US", "CH-US"),       # United States
    ("CH", "JA", "CH-JA"),       # Japan
    ("CH", "AS", "CH-AS"),       # Australia (FIPS code is AS)
    ("CH", "FR", "CH-FR"),       # France
    ("CH", "GM", "CH-GM"),       # Germany
    ("CH", "UK", "CH-UK"),       # United Kingdom
    ("CH", "RS", "CH-RS"),       # Russia
    ("CH", "IN", "CH-IN"),       # India
    ("CH", "ID", "CH-ID"),       # Indonesia
    ("CH", "PK", "CH-PK"),       # Pakistan
    ("CH", "VM", "CH-VM"),       # Vietnam
]

# -----------------------------------------
# 2. KEYWORD LIST (Caldara & Iacoviello, 2022)
# -----------------------------------------
gpr_keywords = [
    "war", "wars", "warfare", "military", "armed conflict", "conflict", 
    "terror", "terrorism", "terrorist", "attack", "attacks", "invasion", 
    "invade", "battle", "troops", "sanction", "sanctions", "embargo", 
    "hostage", "kidnap", "assassination", "coup", "rebel", "insurgent", 
    "nuclear", "missile", "airstrike", "bombing", "combat", "blockade", 
    "geopolitical", "tension", "crisis", "hostile", "retaliation"
]

# -------------------------------
# 3. DATE RANGE (1990–2022)
# -------------------------------
start_date = datetime(1990, 1, 1)
end_date   = datetime(2022, 2, 1)

# Store results in a list
results = []

# -------------------------------
# 4. MONTHLY LOOP (with resume capability)
# -------------------------------
# We'll iterate month by month. 
# If you have an existing progress file, load it to skip months already processed.
processed_months = set()
try:
    prev = pd.read_csv("gdelt_progress.csv", index_col=0, parse_dates=True).index
    processed_months = set(prev.strftime('%Y-%m')) # track by YYYY-MM string
    print(f"Resuming: {len(processed_months)} months already processed.")
except FileNotFoundError:
    print("Starting fresh: no previous progress file found.")

current = start_date
while current <= end_date:
    month_str = current.strftime("%Y-%m")
    
    # skip if already done
    if month_str in processed_months:
        print(f"Skipping {month_str} (already processed)")
        current += timedelta(days=32)
        current = current.replace(day=1)
        continue
    
    print(f"Processing {month_str}...")
    # Prepare date boundaries for the full month
    month_start = current.strftime("%Y-%m-%d")
    next_month = current + timedelta(days=32)
    next_month = next_month.replace(day=1)
    month_end   = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")
    
    row = {"date": current}
    gd = GdeltDoc()
    
    # For each dyad, get GPR shares for both countries
    for code_a, code_b, dyad_name in dyads:
        # --- Country A (China) ---
        try:
            # Articles with GPR keywords
            f1 = Filters(
                start_date = month_start,
                end_date   = month_end,
                country    = [code_a],
                keyword    = gpr_keywords,
                num_records = 250  # maximum allowed; we only need the total count
            )
            gpr_articles = gd.article_search(f1)
            gpr_count = len(gpr_articles) if gpr_articles is not None else 0
            
            # Total articles in that country (no keywords)
            f2 = Filters(
                start_date = month_start,
                end_date   = month_end,
                country    = [code_a],
                num_records = 250
            )
            total_articles = gd.article_search(f2)
            total_count = len(total_articles) if total_articles is not None else 0
            
            if total_count > 0:
                share_a = gpr_count / total_count
            else:
                share_a = 0.0
        except Exception as e:
            print(f"  Error for {code_a} in {month_str}: {e}")
            share_a = None
            total_count = 0
        
        # --- Country B (counterpart) ---
        try:
            f1 = Filters(
                start_date = month_start,
                end_date   = month_end,
                country    = [code_b],
                keyword    = gpr_keywords,
                num_records = 250
            )
            gpr_articles = gd.article_search(f1)
            gpr_count = len(gpr_articles) if gpr_articles is not None else 0
            
            f2 = Filters(
                start_date = month_start,
                end_date   = month_end,
                country    = [code_b],
                num_records = 250
            )
            total_articles = gd.article_search(f2)
            total_count = len(total_articles) if total_articles is not None else 0
            
            if total_count > 0:
                share_b = gpr_count / total_count
            else:
                share_b = 0.0
        except Exception as e:
            print(f"  Error for {code_b} in {month_str}: {e}")
            share_b = None
        
        # Bilateral index = average of the two country‑level GPR shares
        if share_a is not None and share_b is not None:
            bilateral_index = (share_a + share_b) / 2
        else:
            bilateral_index = None
        
        row[f"{dyad_name}_GPR"] = bilateral_index
        row[f"{dyad_name}_total_A"] = total_count  # optional, for diagnostics
    
    # Append result and save progress
    results.append(row)
    
    # Save progress after each month (so you can resume if interrupted)
    pd.DataFrame(results).to_csv("gdelt_gpr_index_monthly.csv", index=False)
    
    # Record that this month is done
    processed_months.add(month_str)
    pd.DataFrame(index=pd.DatetimeIndex([current]), columns=["done"]).to_csv(
        "gdelt_progress.csv", mode='a', header=False
    )
    
    # Move to next month (API rate limit safeguard)
    time.sleep(random.uniform(1.5, 3.5))
    current = next_month

print("Processing complete!")

# -------------------------------
# 5. POST-PROCESSING: TURNING POINTS
# -------------------------------
df = pd.read_csv("gdelt_gpr_index_monthly.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

def second_difference(series):
    """Second difference: Δ²x_t = x_t - 2*x_{t-1} + x_{t-2}"""
    return series - 2*series.shift(1) + series.shift(2)

for dyad_name in [d[2] for d in dyads]:
    col = f"{dyad_name}_GPR"
    if col in df.columns:
        df[f"d2_{col}"] = second_difference(df[col])

# Save final output
df.to_csv("gdelt_gpr_index_final.csv")
print("Final file saved: gdelt_gpr_index_final.csv")