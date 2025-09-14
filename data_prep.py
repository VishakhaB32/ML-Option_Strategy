import pandas as pd
import polars as pl


def load_spot_data(path="data/spot_with_signals_2023.csv"):
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["datetime"] = df["datetime"].dt.tz_convert(None)   # tz-naive
    df["closest_expiry"] = pd.to_datetime(df["closest_expiry"], errors="coerce")  # ✅ force datetime
    return df



def load_options_data(path="data/options_data_2023.parquet"):
    """
    Load options data using Polars (lazy).
    Ensures datetime columns are timezone-naive.
    """
    lf = pl.scan_parquet(path, use_statistics=True)
    
    # make datetime tz-naive
    lf = lf.with_columns([
        pl.col("datetime").dt.replace_time_zone(None),
        pl.col("expiry_date").dt.replace_time_zone(None)
    ])
    
    return lf
