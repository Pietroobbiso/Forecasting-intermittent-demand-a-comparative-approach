import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from datetime import timedelta
import altair as alt
import matplotlib.dates as mdates

from sklearn.mixture import GaussianMixture
from sktime.utils.plotting import plot_series

from process_analytics.settings.settings import bdf

os.chdir(r"C:\Users\39327\Documents\GitHub\process-analytics\process_analytics")
#print(os.getcwd())
'''
# DATA VISUALIZATION FOR SCENARIOS
data_to_plot = bdf.set_index(bdf.event_starts, inplace=False)
#uncomment one each time
#data_to_plot = data_to_plot.loc["2020-11-29":"2020-12-5", :]
#data_to_plot = data_to_plot.loc["2020-12-22":"2020-12-28", :]
data_to_plot = data_to_plot.loc["2020-12-25":"2021-01-01", :]
#uncomment one each time
#data_to_plot_in_red = data_to_plot.loc["2020-12-01 00:00:00+01:00 ":"2020-12-02 23:00:00+01:00", :]
#data_to_plot_in_red = data_to_plot.loc["2020-12-01 12:00:00+01:00 ":"2020-12-03 11:00:00+01:00", :]
#data_to_plot_in_red = data_to_plot.loc["2020-12-24 00:00:00+01:00 ":"2020-12-25 23:00:00+01:00", :]
#data_to_plot_in_red = data_to_plot.loc["2020-12-24 12:00:00+01:00 ":"2020-12-26 11:00:00+01:00", :]
#data_to_plot_in_red = data_to_plot.loc["2020-12-29 00:00:00+01:00 ":"2020-12-30 23:00:00+01:00", :]
data_to_plot_in_red = data_to_plot.loc["2020-12-29 12:00:00+01:00 ":"2020-12-31 11:00:00+01:00", :]

print(data_to_plot.tail())
print(data_to_plot_in_red)
data_to_plot = data_to_plot.squeeze()
data_to_plot_in_red = data_to_plot_in_red.squeeze()
fig, ax = plt.subplots(1, figsize=plt.figaspect(.4))
data_to_plot.plot(ax=ax)
data_to_plot_in_red.plot(color = 'red')
ax.set(ylabel=data_to_plot.name)
ax.set_ylabel('Value')
ax.set_xlabel('Day')
ax.set_title('Scenario 6')
plt.show()
'''
#Histogram of values
bdf1 = bdf.set_index(bdf.event_starts, inplace=False)
print(bdf1.value_counts())
fig, ax = plt.subplots(1, figsize=plt.figaspect(.4))
plt.style.use('ggplot')
plt.hist(bdf1,bins=200)
plt.xlim(0,1800)
#ax.set_xticks([0,2000])
#ax.set(ylabel=data_to_plot.name)
ax.set_ylabel('Count')
ax.set_xlabel('Values (liters)')
plt.show()

'''
# Average Water consumption Over Hour, Day, Week, Month
fig = plt.figure(figsize=(18, 16))
fig.subplots_adjust(hspace=0.9)
ax1 = fig.add_subplot(4, 1, 1)
ax1.plot(bdf1["event_value"].resample("H").mean(), linewidth=1)
ax1.set_title("Water consumption over hour")
ax1.tick_params(axis="both", which="major")


ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
ax2.plot(bdf1["event_value"].resample("D").mean(), linewidth=1)
ax2.set_title("Water consumption over day")
ax2.tick_params(axis="both", which="major")


ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
ax3.plot(bdf1["event_value"].resample("W").mean(), linewidth=1)
ax3.set_title("Water consumption over week")
ax3.tick_params(axis="both", which="major")
ax3.set_xlim([25, 50])


ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
ax4.plot(bdf1["event_value"].resample("M").mean(), linewidth=1)
ax4.set_title("Water consumption over month")
ax4.tick_params(axis="both", which="major")
plt.show()
'''
bdf["year"] = bdf.event_starts.year
bdf["month"] = bdf.event_starts.month
bdf["day"] = bdf.event_starts.day
bdf["quarter"] = bdf.event_starts.quarter
bdf["week"] = bdf.event_starts.week
#print(bdf["week"].values)
bdf["hour"] = bdf.event_starts.hour
bdf["weekday"] = bdf.event_starts.dayofweek
bdf["weekend"] = (bdf["weekday"] >= 5).astype(int)  # 1if weekend, 0 if weekday
#print(bdf[bdf["weekday"] == 3])


# Water consumption Weekdays vs. Weekends
dic = {0: "Weekend", 1: "Weekday"}
bdf["Day"] = bdf.weekday.map(dic)

# # Box Plot of Water consumption by Weekday vs. Weekend per month
# a = plt.figure(figsize=(9, 4))
# plt1 = sns.boxplot("day", "event_value", hue="Day", width=0.6, fliersize=3, data=bdf)
# # a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
# sns.despine(left=True, bottom=True)
# plt.title("Box Plot of Water consumption by Weekday vs. Weekend per month")
# plt.xlabel("")
# plt.tight_layout()
# plt.show()
#
# # Box Plot of Water consumption by Weekday vs. Weekend per hour
# #a = plt.figure(figsize=(9, 4))
# plt2 = sns.boxplot("week", "event_value", hue="Day", width=0.6, fliersize=3, data=bdf)
# a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
# sns.despine(left=True, bottom=True)
# plt.title("Box Plot of Water consumption by Weekday vs. Weekend per hour")
# plt.xlabel("")
# plt.tight_layout()
# plt.show()

# Factor Plot of Water consumption by Weekday vs. Weekend per hour
plt3 = sns.factorplot(
    "hour", "event_value", hue="Day", data=bdf, kind = 'strip', size=4, aspect=1.5, legend=False
)
plt.title("Factor Plot of Water consumption by Weekend/Weekday per hour")
plt.tight_layout()
sns.despine(left=True, bottom=True)
plt.legend(loc="upper right")
plt.show()
'''
# Factor Plot of Water consumption by Weekday vs. Weekend per month
plt4 = sns.factorplot(
    "month", "event_value", hue="Day", data=bdf, size=4, aspect=1.5, legend=False
)
plt.title("Factor Plot of Water consumption by Weekend/Weekday per month")
plt.tight_layout()
sns.despine(left=True, bottom=True)
plt.legend(loc="upper right")
plt.show()



# Factor Plot of Water consumption by Weekday vs. Weekend per day of the week
plt5 = sns.factorplot(
    "weekday", "event_value", hue="Day", data=bdf, size=4, aspect=1.5, legend=False
)
plt.title("Factor Plot of Water consumption by Weekend/Weekday per day of the week")
plt.tight_layout()
sns.despine(left=True, bottom=True)
plt.legend(loc="upper right")
plt.show()



#Box Plot of Yearly vs. Quarterly Global Active Power
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.subplots_adjust(wspace=0.2)
sns.boxplot(x="hour", y="event_value", data=bdf)
plt.xlabel("hour")
plt.title("Box plot of Hourly Global Active Power")
sns.despine(left=True)
plt.tight_layout()
plt.subplot(1, 2, 2)
sns.boxplot(x="month", y="event_value", data=bdf)
plt.xlabel("month")
plt.title("Box plot of Monthly Global Active Power")
sns.despine(left=True)
plt.tight_layout()
plt.show()


# Perform cluster analysis (1)
weekly_total = bdf.groupby("week").mean()
weekly_total = weekly_total["event_value"]
weekly_total = np.array(weekly_total).reshape(-1, 1)
print(weekly_total)
print(len(weekly_total))
numbers = range(1, 10)
sequence_of_numbers = [number for number in numbers if (number % 5 in (1, 2))]

daily_total = bdf.groupby("day").mean()
print(daily_total["event_value"])

weekday_total = bdf.groupby("weekday").mean()
print(weekday_total["event_value"])

hourly_total = bdf.groupby("hour").mean()
hourly_total = hourly_total["event_value"]
hourly_total = np.array(hourly_total).reshape(-1, 1)
print(hourly_total)
'''
# USE ALTAIR PACKAGE
alt.data_transformers.disable_max_rows()

# Interval Selection altair
#Data to plot
source = bdf.reset_index()
source["belief_horizon"] = source["belief_horizon"] / timedelta(seconds=1)
source["source"] = source["source"].apply(lambda x: x.name)

brush = alt.selection(type="interval", encodings=["x"])
base = (
    alt.Chart(source)
    .mark_bar()
    .encode(
        x="event_start:T",
        # y = 'event_value:Q'
        tooltip=[
            alt.Tooltip(
                "event_start:T",
                timeUnit="yearmonthdatehoursminutes",
                title="Event start",
            ),
            alt.Tooltip("event_value:Q", title="Event value"),
        ],
    )
    .properties(width=8000, height=400)
)

upper = (
    base.encode(
        alt.X("event_start:T", scale=alt.Scale(domain=brush)), alt.Y("event_value:Q")
    )
    .transform_filter(brush)
    .add_selection(alt.selection_single())
)  # the upper part of the graph results in scaled y-axis limit

lower = base.encode(alt.Y("event_value:Q")).properties(height=60).add_selection(brush)

chart = upper & lower  # + symbol for layer another chart on top of these

chart.show()

# GAUSSIAN MIXTURE MODELS FOR DAILY_TOTAL

# Set up a range of cluster numbers to try
n_range = range(2, 11)

# Create empty lists to store the BIC and AIC values
bic_score = []
aic_score = []

# Loop through the range and fit a model
for n in n_range:
    gm = GaussianMixture(n_components=n, random_state=123, n_init=10)
    gm.fit(daily_total)

# Append the BIC and AIC to the respective lists
bic_score.append(gm.bic(daily_total))
aic_score.append(gm.aic(daily_total))

# Plot the BIC and AIC values together
fig, ax = plt.subplots(figsize=(12, 8), nrows=1)
ax.plot(n_range, bic_score, "-o", color="orange")
ax.plot(n_range, aic_score, "-o", color="green")
ax.set(xlabel="Number of Clusters", ylabel="Score")
ax.set_xticks(n_range)
ax.set_title("BIC and AIC Scores Per Number Of Clusters")
plt.show()

gm = GaussianMixture(n_components=5, random_state=123, n_init=10)
preds = gm.fit_predict(daily_total)
print(preds)

# GAUSSIAN MIXTURE MODELS FOR WEEKLY_TOTAL

# Set up a range of cluster numbers to try
n_range = range(2, 11)

# Create empty lists to store the BIC and AIC values
bic_score = []
aic_score = []

# Loop through the range and fit a model
for n in n_range:
    gm = GaussianMixture(n_components=n, random_state=123, n_init=10)
    gm.fit(weekly_total)

    # Append the BIC and AIC to the respective lists
    bic_score.append(gm.bic(weekly_total))
    aic_score.append(gm.aic(weekly_total))

# Plot the BIC and AIC values together
fig, ax = plt.subplots(figsize=(12, 8), nrows=1)
ax.plot(n_range, bic_score, "-o", color="orange")
ax.plot(n_range, aic_score, "-o", color="green")
ax.set(xlabel="Number of Clusters", ylabel="Score")
ax.set_xticks(n_range)
ax.set_title("BIC and AIC Scores Per Number Of Clusters")
plt.show()

gm = GaussianMixture(n_components=5, random_state=123, n_init=10)
preds = gm.fit_predict(weekly_total)
print(preds)

# SPLIT THE DATA ACCORDING TO APRIL 2020


april20_time = pd.Timestamp("2020-04-01 00:00:00+02:00").astimezone("Europe/Amsterdam")
bdf_bef_apr20 = bdf[bdf.index.get_level_values("event_start") < april20_time]
bdf_aft_apr20 = bdf[bdf.index.get_level_values("event_start") >= april20_time]
print(bdf_bef_apr20)
print(bdf_aft_apr20)

# FactorPlot of Water consumption by Weekday vs.Weekend per hour(data before April 2020)
plt3 = sns.factorplot(
    "hour",
    "event_value",
    hue="Day",
    data=bdf_bef_apr20,
    size=4,
    aspect=1.5,
    legend=False,
)
plt.title(
    "Factor Plot of Water consumption by Weekend/Weekday"
    " per hour (data before April 2020)"
)
plt.tight_layout()
sns.despine(left=True, bottom=True)
plt.legend(loc="upper right")
plt.show()

# Factor Plot of Water consumption by Weekday vs.Weekend
# per hour(data after April 2020) #check it if mean (better using the median)
plt4 = sns.factorplot(
    "hour",
    "event_value",
    hue="Day",
    data=bdf_aft_apr20,
    size=4,
    aspect=1.5,
    legend=False,
)
plt.title(
    "Factor Plot of Water consumption by Weekend/Weekday"
    "per hour (data after April 2020)"
)
plt.tight_layout()
sns.despine(left=True, bottom=True)
plt.legend(loc="upper right")
plt.show()
