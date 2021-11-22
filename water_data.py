# noqa E501
import os
from datetime import timedelta

import pandas as pd
import timely_beliefs as tb

from process_analytics.models import multiple_output_strategies
from process_analytics.models import direct_multistep_strategies
from process_analytics.models import sktime
from process_analytics.settings.settings import (
    bdf,
    current_time,
    n_events,
    reference_source,
)

#print(os.getcwd())


# Load data
#print(f"bdf = {bdf}")

# All measurements known at current_time
# (note that the event starting at current time is not known yet)
# current_bdf = bdf.fixed_viewpoint(current_time)  # too slow
current_bdf = bdf[bdf.index.get_level_values("event_start") < current_time]
#print(f"current_bdf = {current_bdf}")

# realised_bdf = bdf[
#     (bdf.index.get_level_values("event_start") >= current_time)
#     & (
#         bdf.index.get_level_values("event_start")
#         < current_time + timedelta(hours=n_events)
#     )
# ]
# print(f"realised_bdf = {realised_bdf}")

# (any forecaster should only use current_bdf, because bdf contains future values, too)

# naive_forecast_bdf = multiple_output_strategies.naive_forecaster(
#     current_bdf, current_time, FREQ='1H',MAX_FORECAST_HORIZON_HOURS=48)
# print(naive_forecast_bdf)

# lr_forecast_bdf = direct_multistep_strategies.linear_regression_forecaster(
#     current_bdf, current_time, n_events
# )
#
# rf_forecast_bdf = direct_multistep_strategies.random_forest_forecaster(
#     current_bdf, current_time, n_events
# )
#
# dt_forecast_bdf = direct_multistep_strategies.decision_tree_forecaster(
#     current_bdf, current_time, n_events
# )
#
# knn_forecast_bdf = direct_multistep_strategies.k_nearest_neighbours_forecaster(
#     current_bdf, current_time, n_events
# )
# #
# svr_forecast_bdf = direct_multistep_strategies.support_vector_machine_forecaster(
#     current_bdf, current_time, n_events
# )

# poi_reg_forecast_bdf = direct_multistep_strategies.poisson_regression_forecaster(
# current_bdf, current_time, n_events
# )

naive_forecast_bdf = sktime.naive_forecaster(
    current_bdf, current_time, bdf, n_events
)
naive_forecast_with_seas_bdf = sktime.naive_forecaster_with_seasonality(  # noqa E501
    current_bdf, current_time, bdf, n_events
)
exp_smoothing = sktime.exponential_smoothing(
    current_bdf, current_time, bdf, n_events
)
knn_forecast_bdf = sktime.knearest_neighbors(
    current_bdf, current_time, bdf, n_events
)
AutoETS_forecast_bdf = sktime.AutoETS_forecaster(
    current_bdf, current_time, bdf, n_events
)

# All measurements for the next n_events
realised_bdf = bdf[
    (bdf.index.get_level_values("event_start") >= current_time)
    & (
        bdf.index.get_level_values("event_start")
        < current_time + timedelta(hours=n_events)
    )
]
print(f"realised_bdf = {realised_bdf}")

# Combine forecasts and realised values (sktime)
results_bdf: tb.BeliefsDataFrame = pd.concat(
    [
        naive_forecast_with_seas_bdf,
        exp_smoothing,
        knn_forecast_bdf,
        AutoETS_forecast_bdf,
        realised_bdf
    ]
).sort_index()

# Combine forecasts and realised values (sklearn)
# results_bdf: tb.BeliefsDataFrame = pd.concat(
#     [
#         naive_forecast_bdf,
#         # lr_forecast_bdf,
#         # dt_forecast_bdf,
#         # rf_forecast_bdf,
#         # knn_forecast_bdf,
#         # svr_forecast_bdf,
#         realised_bdf
#     ]
# ).sort_index()
# print(f"results_bdf = {results_bdf}")

# Score the forecasts with the realised values as their reference
scores = results_bdf.fixed_viewpoint_accuracy(
    belief_time=current_time, reference_source=reference_source
)
print(f"scores = \n{scores}")

# Serve an interactive dashboard of the results on a local port
chart = results_bdf.plot(
    show_accuracy=True,
    reference_source=reference_source,
    intuitive_forecast_horizon=False,
)
chart.serve()
