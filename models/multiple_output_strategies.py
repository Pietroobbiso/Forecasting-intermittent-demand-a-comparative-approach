import numpy as np
import pandas as pd
import timely_beliefs as tb

from process_analytics.utils import data_utils
from process_analytics.utils import forecast_utils

def naive_forecaster(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    FREQ: str,
    MAX_FORECAST_HORIZON_HOURS: int,
) -> tb.BeliefsDataFrame:
    """Forecaster: naive model."""
    naive_forecaster = tb.BeliefSource("Naive forecaster")
    forecast_input = current_bdf.xs(
        current_time - pd.Timedelta(FREQ), level="event_start", drop_level=False
    )  # Latest measurement known at current time
    # probably equivalent to the line above, but needs index to be sorted by event_start
    # forecast_input = current_bdf.tail(1)
    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=np.repeat(
            forecast_input["event_value"].values, MAX_FORECAST_HORIZON_HOURS
        ),
        sensor=current_bdf.sensor,
        forecaster=naive_forecaster,
        current_time=current_time,
    )
    return forecast_bdf
