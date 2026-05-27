import pandas as pd
import gdelt

print("=" * 60)
print("CHECKING GDELT CSV STRUCTURE ACROSS TIME BLOCKS")
print("=" * 60)

gd = gdelt.gdelt(version=1)

# Check a sample month every 5 years
sample_years = ['1990 Jan', '1995 Jan', '2000 Jan', '2005 Jan', '2010 Jan', '2015 Jan', '2020 Jan']

for sample in sample_years:
    print(f"\nFetching columns for sample: {sample}...")
    try:
        df = gd.Search(sample, table='events', coverage=True)
        if df is not None and not df.empty:
            # Clean column names
            df.columns = df.columns.str.strip()
            print(f"  -> SUCCESS! Found {len(df.columns)} columns.")
            print(f"  -> Core columns check:")
            
            # Check if our target columns exist
            for col in ['Actor1CountryCode', 'Actor2CountryCode', 'GoldsteinScale']:
                status = "✅ Present" if col in df.columns else "❌ MISSING"
                print(f"     - {col}: {status}")
        else:
            print("  -> Returned empty DataFrame.")
    except Exception as e:
        print(f"  -> ❌ Error fetching this block: {e}")

print("\n" + "=" * 60)