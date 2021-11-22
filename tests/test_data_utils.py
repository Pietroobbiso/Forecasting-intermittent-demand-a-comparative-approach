from datetime import datetime, timedelta
import pytz

import numpy as np
import pandas as pd
import pytest
import timely_beliefs as tb

from process_analytics.tests._test_utils import equal_arrays
from process_analytics.utils.forecast_utils import forecasts_to_beliefs


@pytest.mark.parametrize(
    "forecasts",
    [[10, 11, 10], [10, 11, 10.4], np.array([10, 11, 10]), np.array([10, 11, 10.4])],
)
def test_forecasts_to_beliefs(forecasts):
    """Check whether forecasts_to_beliefs returns a BeliefsDataFrame,
    with the expected event starts, horizons and values.
    """
    sensor = tb.Sensor(
        "Some sensor", unit="Some unit", event_resolution=timedelta(hours=1)
    )
    source = tb.BeliefSource("Some source")
    current_time = datetime(2021, 3, 5, tzinfo=pytz.utc)
    bdf = forecasts_to_beliefs(forecasts, sensor, source, current_time)
    assert isinstance(bdf, tb.BeliefsDataFrame)
    assert equal_arrays(
        bdf.event_starts.values,
        pd.date_range(
            current_time,
            current_time + sensor.event_resolution * len(forecasts),
            closed="left",
        ).values,
    )
    assert equal_arrays(
        bdf.belief_horizons.to_pytimedelta(),
        [sensor.event_resolution * (h + 1) for h in range(len(forecasts))],
    )
    assert equal_arrays(bdf.values, forecasts)
