from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "fuel_lvl", "speed", "contact_value", "gps_latitude", "gps_longitude",
    "fuel_lvl_change", "fuel_lvl_change_rate", "is_stationary", "is_weekend",
    "previous_fuel_lvl", "previous_speed", "distance_traveled",
    "rolling_mean_fuel_change_5min", "rolling_mean_speed_5min",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "day_of_month_sin", "day_of_month_cos", "month_sin", "month_cos"
]


def haversine_distance(lat1, lon1, lat2, lon2):
    if lat2 is None or lon2 is None or pd.isna(lat2) or pd.isna(lon2):
        return 0.0

    r = 6371  # km

    lat1, lon1, lat2, lon2 = map(
        radians,
        [float(lat1), float(lon1), float(lat2), float(lon2)]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return r * c


def parse_time(value):
    if isinstance(value, datetime):
        return value
    return pd.to_datetime(value).to_pydatetime()


def build_features(current_measurement: dict, history: list[dict] | None = None) -> dict:
    """
    current_measurement = trenutno očitanje vozila
    history = prethodna očitanja istog vozila, idealno zadnjih barem 5 minuta
    """

    history = history or []

    rows = history + [current_measurement]
    df = pd.DataFrame(rows)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["fuel_lvl_change"] = df["fuel_lvl"].diff()
    df["time_diff"] = df["time"].diff().dt.total_seconds()

    df["fuel_lvl_change_rate"] = df["fuel_lvl_change"] / df["time_diff"]
    df["fuel_lvl_change_rate"] = df["fuel_lvl_change_rate"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df["is_stationary"] = (df["speed"] == 0).astype(int)

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["day_of_month"] = df["time"].dt.day
    df["month"] = df["time"].dt.month

    df["previous_fuel_lvl"] = df["fuel_lvl"].shift(1)
    df["previous_speed"] = df["speed"].shift(1)

    df["prev_gps_latitude"] = df["gps_latitude"].shift(1)
    df["prev_gps_longitude"] = df["gps_longitude"].shift(1)

    df["distance_traveled"] = df.apply(
        lambda row: haversine_distance(
            row["gps_latitude"],
            row["gps_longitude"],
            row["prev_gps_latitude"],
            row["prev_gps_longitude"]
        ),
        axis=1
    )

    df["fuel_lvl_change"] = df["fuel_lvl_change"].fillna(0)
    df["time_diff"] = df["time_diff"].fillna(0.0)
    df["fuel_lvl_change_rate"] = df["fuel_lvl_change_rate"].fillna(0)
    df["previous_fuel_lvl"] = df["previous_fuel_lvl"].fillna(0)
    df["previous_speed"] = df["previous_speed"].fillna(0)

    df = df.set_index("time")

    df["rolling_mean_fuel_change_5min"] = (
        df["fuel_lvl_change"]
        .rolling(window="5min", min_periods=1)
        .mean()
    )

    df["rolling_mean_speed_5min"] = (
        df["speed"]
        .rolling(window="5min", min_periods=1)
        .mean()
    )

    df = df.reset_index()

    cyclical_features = {
        "hour": 24,
        "day_of_week": 7,
        "day_of_month": 31,
        "month": 12
    }

    for col, period in cyclical_features.items():
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    latest = df.iloc[-1].to_dict()

    return {
        col: float(latest[col])
        for col in FEATURE_COLUMNS
    }