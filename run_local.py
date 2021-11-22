import pandas as pd

from process_analytics.utils import plotting_utils, data_utils

# Analysis settings
customer_id = "2085935"
current_time = pd.Timestamp("2019-10-10 10:00:00+02:00")

# Config settings
FREQ = "1H"
AGG_DEMAND_UNIT = "m³"
# HEAD
DATA_PATH = f"data/HTC forecasting benchmark/{customer_id}_{FREQ}/analysis.csv"
PLOT_PATH = f"data/HTC forecasting benchmark/{customer_id}_{FREQ}/analysis.html"


# Add some plotting utils and scoring utils, needing clean-up.

# Load data
df = data_utils.load_process_data(DATA_PATH)
# print(df)

# Add a column with the start time of the most recent hour
df["previous_belief_time"] = df["belief_time"] - pd.Timedelta(FREQ)

# Assuming a belief time, filter out past and present active processes
# (for plotting the decomposition of the original time series)
df_plot = df.copy()[df["active"] & (df["belief_time"] <= current_time)]

print(df_plot)

# Create plot (in html file)
df_plot = plotting_utils.altairify(df_plot)
chart = plotting_utils.process_bar_chart(df_plot, AGG_DEMAND_UNIT)
html = plotting_utils.from_html_template(chart, customer_id)
plotting_utils.save_html(html, PLOT_PATH)

# Assuming a belief time, filter out past finished processes
# (for creating a sample to estimate a multivariate distribution)
df_sample = df.copy()[(~df["active"] & (df["belief_time"] < current_time))]

# Assuming a belief time, filter out currently active processes
# (when are they likely to end? And what processes are likely to
# start within the next 48 hours?)
df_current = df.copy()[df["active"] & (df["belief_time"] == current_time)]
