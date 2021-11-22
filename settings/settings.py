from datetime import timedelta

import pandas as pd
import os

# Analysis settings
import timely_beliefs as tb

from process_analytics.utils import data_utils

os.chdir(r"C:\Users\39327\Documents\GitHub\process-analytics\process_analytics")
#print(os.getcwd())

customer_id = "2085930"

#Relevant dates: 3 dates in the same timezone characteristics (01 or 02), choose outside of the summertime (+01), at the end of 2020, one time which is at midnight and one at noon,
#1) 2020-12-01 00:00 -> a moment of a very low demand at night of a normal week (write which day of the week it was) for every date below
#2) 2020-12-01 12:00 -> a moment of non zero demand of a mid day of a normal week
#3) 2020-12-29 00:00 -> a moment at night in a week into the anomaly (predict tuesday and wednesday)
#4) 2020-12-29 12:00 -> a moment at midday in a week into the anomaly (the forecast will include the end of the anomaly, i.e. 31 Dec 8am/10am)
#last moment to pick is 30 December at 00:00
#5) 2020-12-24 12:00 -> Day of when the anomaly starts at noon
#6) 2020-12-24 00:00 -> Day of when the anomaly starts at midnight

#until 25 october of 2020 -> summertime -> +02, then +01
current_time = pd.Timestamp("2020-12-29 12:00:00+01:00").astimezone("Europe/Amsterdam") #Anomaly is: 24th December 2020
# Config settings
FREQ = "1H"
AGG_DEMAND_UNIT = "m³"
DATA_PATH = f"../data/HTC forecasting benchmark/{customer_id}_20210101111913.csv"
PLOT_PATH = f"../data/HTC forecasting benchmark/{customer_id}_{FREQ}/analysis.html"
n_events = 48
reference_source = tb.BeliefSource("Water company")


def load_data_as_source(DATA_PATH, FREQ, customer_id, AGG_DEMAND_UNIT):
    s = data_utils.load_time_series_data(DATA_PATH, FREQ)
    bdf = tb.BeliefsDataFrame(
        s,
        sensor=tb.Sensor(
            name=f"Water meter {customer_id}",
            unit=data_utils.determine_k_unit_str(AGG_DEMAND_UNIT),
            event_resolution=pd.Timedelta(FREQ).to_pytimedelta(),
        ),
        source=reference_source,
        belief_horizon=timedelta(0),
    )
    return bdf


bdf = load_data_as_source(DATA_PATH, FREQ, customer_id, AGG_DEMAND_UNIT)
#bdf = bdf.convert_index_from_belief_horizon_to_time()
print(bdf.head())