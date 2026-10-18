
import pandas as pd
from pybaseball import statcast
from pathlib import Path

# 1. Configuration
months = pd.period_range("2024-03", "2025-09", freq="M")
CACHE_FILE = Path(__file__).resolve().parent / "statcast_raw.pkl"

# 2. Download or load cached data
if CACHE_FILE.exists():
    print("Loading cached Statcast data...")
    data = pd.read_pickle(CACHE_FILE)
else:
    chunks = []

    for month in months:
        start = max(month.start_time, pd.Timestamp("2024-03-01"))
        end = min(month.end_time, pd.Timestamp("2025-09-30"))

        print(f"Downloading {month}...")
        chunk = statcast(
            start_dt=start.strftime("%Y-%m-%d"),
            end_dt=end.strftime("%Y-%m-%d")
        )
        chunks.append(chunk)

    data = pd.concat(chunks, ignore_index=True)

    # Cache ALL raw pitch data
    data.to_pickle(CACHE_FILE)
    print("Saved raw data!")

# 3. Inspect raw data
print("Rows:", len(data))
print("Columns:", len(data.columns))
print(data.head())
