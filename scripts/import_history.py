import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

DATA_DIR = Path("data")
CSV_FILES = list(DATA_DIR.glob("*.csv"))

DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)

dfs = []

for file in CSV_FILES:
    df_part = pd.read_csv(file, sep=";")
    df_part.columns = df_part.columns.str.strip()
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

df["time"] = pd.to_datetime(df["time"])

df["obj_id"] = df["obj_id"].astype(str)
df["fuel_lvl"] = pd.to_numeric(df["fuel_lvl"], errors="coerce")
df["speed"] = pd.to_numeric(df["speed"], errors="coerce")

contact_mapping = {
    "-": 0,
    "Neaktivan": 0,
    "Aktivan": 1
}

df["contact_value"] = (
    df["contact_value"]
    .map(contact_mapping)
    .fillna(0)
    .astype(int)
)

df["gps_latitude"] = (
    df["gps_latitude"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

df["gps_longitude"] = (
    df["gps_longitude"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

median_latitude = df["gps_latitude"].median()
median_longitude = df["gps_longitude"].median()

df["gps_latitude"] = df["gps_latitude"].fillna(median_latitude)
df["gps_longitude"] = df["gps_longitude"].fillna(median_longitude)

df = df.sort_values(["obj_id", "time"])

test_last_rows = df.groupby("obj_id").tail(1)
history_rows = df.drop(test_last_rows.index)

columns = [
    "obj_id",
    "time",
    "fuel_lvl",
    "speed",
    "contact_value",
    "gps_latitude",
    "gps_longitude"
]

history_rows[columns].to_sql(
    name="vehicle_measurements",
    con=engine,
    schema="fuel_theft",
    if_exists="append",
    index=False
)

test_last_rows[columns].to_csv("data/test_last_rows.csv", index=False)

print(f"CSV files loaded: {len(CSV_FILES)}")
print(f"Inserted rows: {len(history_rows)}")
print(f"Saved test rows: {len(test_last_rows)}")