import pandas as pd

from process_analytics.defaults import K_TIME_UNIT_STR


def load_process_data(path: str) -> pd.DataFrame:
    """Load process data and make the data types nice to work with."""
    df = pd.read_csv(path, infer_datetime_format=True, parse_dates=["belief_time"])
    for dt_col in ["belief_time", "s", "e"]:
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True).dt.tz_convert(
            "Europe/Amsterdam"
        )  # ensure datetime columns have dtype='datetime64[ns, Europe/Amsterdam]
    df["active"] = df["active"].astype("bool")  # ensure active column has dtype='bool'
    return df


def load_time_series_data(path: str, freq_str: str) -> pd.Series:
    """Load time series data and make the data types nice to work with."""
    df = pd.read_csv(
        path,
        sep=";",
        infer_datetime_format=True,
        dayfirst=True,
        parse_dates=["Timestamp"],
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert(
        "Europe/Amsterdam"
    )  # ensure datetime columns have dtype='datetime64[ns, Europe/Amsterdam]
    df = df.set_index("Timestamp")
    # Resample data
    s = (
        df.resample(freq_str, label="right", closed="right")
        .last()
        .diff(1)
        .shift(-2)
        .rename(columns={"Value": "event_value"})["event_value"][:-2]
    )  # index is start time of 1-hour time slot
    s.index = s.index.rename("event_start")

    # Return the last sequence without NaN values
    first_index_of_sequence_without_nan_values = s.loc[pd.isna(s)].index[-1]
    s = s[s.index > first_index_of_sequence_without_nan_values]
    return s


def determine_k_unit_str(agg_demand_unit: str):
    """For example:
    >>> calculate_k_unit_str("m³")  # m³/h
    >>> calculate_k_unit_str("kWh")  # kW
    """
    k_unit_str = (
        agg_demand_unit.rpartition(K_TIME_UNIT_STR)[0]
        if agg_demand_unit.endswith(K_TIME_UNIT_STR)
        else f"{agg_demand_unit}/{K_TIME_UNIT_STR}"
    )
    return k_unit_str
