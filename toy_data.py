import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

# from sktime.forecasting.ets import AutoETS
# from sktime.forecasting.bats import BATS
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series


pd.options.mode.chained_assignment = None


index = pd.date_range(start="2000-1-1", end="2001-1-1", freq="H", closed="left")
random_data = np.random.random_sample((len(index), 1))
noise = 50  # I can play with the noise value
dfs_hourly = []
for i in [0, 4, 15, 70, 35, 84, 50, 70, 20, 90, 43, 100]:
    hourly_i = pd.DataFrame(data=(random_data * 2 - 1) * noise + i, index=index).clip(
        lower=0
    )
    dfs_hourly.append(hourly_i)
#print(dfs_hourly)
# Create the final_df by selecting:
# different time interval differentiating for weekdays and weekend
time = [0, 4, 6, 12, 15, 20, 24]
dictt = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5}
dfs_hourly_weekdays = []
dfs_hourly_weekend = []
for i, j in enumerate(dfs_hourly):
    if i % 2 == 0:
        dfs_hourly_j_weekday = j[
            ((j.index.hour >= time[dictt[i]]) & (j.index.hour < time[dictt[i] + 1]))
            & (j.index.weekday < 5)
        ]
        dfs_hourly_weekdays.append(dfs_hourly_j_weekday)
    else:
        dfs_hourly_j_weekend = j[
            ((j.index.hour >= time[dictt[i]]) & (j.index.hour < time[dictt[i] + 1]))
            & (j.index.weekday >= 5)
        ]
        dfs_hourly_weekend.append(dfs_hourly_j_weekend)
for dataset in dfs_hourly_weekdays:
    dfs_hourly_weekend.append(dataset)
final_df = pd.concat(dfs_hourly_weekend).sort_index()
#print(final_df)

#Toy dataset to be plot
# data_to_plot = final_df.loc["2000-5-15":"2000-5-31", :]
# #print(data_to_plot)
# data_to_plot = data_to_plot.squeeze()
# #print(data_to_plot)
# fig, ax = plt.subplots(1, figsize=plt.figaspect(.4))
# data_to_plot.plot(marker='o',ax=ax,markersize=4)
# #plt.title('')
# ax.set(ylabel=data_to_plot.name)
# ax.set_ylabel('Value')
# ax.set_xlabel('Day')
# # plot_series(data_to_plot)
# #plt.title("Water data - April")
# plt.show()


# Adding anomalies
final_df.loc["2000-12-23":"2000-12-30", :] += 100  # Anomaly in December
final_df.loc["2000-3-1":"2000-3-10", :] += 100  # Anomaly in March
final_df.loc["2000-8-1":"2000-8-15"] += 100  # Anomaly in August

#Plot anomaly
# data_to_plot = final_df.loc["2000-12-15":"2001-1-1", :]
# #print(data_to_plot)
# data_to_plot = data_to_plot.squeeze()
# #print(data_to_plot)
# fig, ax = plt.subplots(1, figsize=plt.figaspect(.4))
# data_to_plot.plot(marker='o',ax=ax,markersize=4)
# #plt.title('')
# ax.set(ylabel=data_to_plot.name)
# ax.set_ylabel('Value')
# ax.set_xlabel('Day')
# # plot_series(data_to_plot)
# #plt.title("Water data - April")
# plt.show()

# Todo: investigate when we add one regressor column with a constant value
# if it works, put useful info into that column, whether it is weekday or weekend
# if it works, add 31 columns (time_of_day and day_of_week)

day_of_week = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
#print(len(day_of_week))
for i, x in enumerate(day_of_week):
    final_df[x] = (final_df.index.get_level_values(0).weekday == i).astype(int)

time_of_day = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
]
for j, y in enumerate(time_of_day):
    final_df[y] = (final_df.index.get_level_values(0).hour == j).astype(int)
#print(final_df)



n_steps = 96
# Define X and y (i.e. Regressors and y_train)
train, test = temporal_train_test_split(final_df, test_size=n_steps)
regressors_train = train[day_of_week + time_of_day]
regressors_test = test[day_of_week + time_of_day]
y_train = train[0]
y_test = test[0]
# print(y_train)
y_train.index = y_train.index.to_period("H")
y_test.index = y_test.index.to_period("H")
regressors_train.index = regressors_train.index.to_period("H")
regressors_test.index = regressors_test.index.to_period("H")

# print(f"regressors_train = \n{regressors_train}")
# print(f"regressors_test = \n{regressors_test}")
# print(f"y_train = \n{y_train}")
# print(f"y_test = \n{y_test}")

fh = ForecastingHorizon(regressors_test.index, is_relative=False)
#print(fh)

# Naïve forecasters (with and without seasonal periodicity)
forecaster_title = ["NaiveForecaster", "NaiveForecaster with seasonal periodicity"]
sp = [1, 7 * 24]
for i, j in zip(sp, forecaster_title):
    forecaster = NaiveForecaster(strategy="last", sp=i)
    forecaster.fit(y=y_train, X=regressors_train)
    y_pred = forecaster.predict(fh)
    #print(y_pred)
    plot_series(
        y_train.tail(n_steps),
        y_test.tail(n_steps),
        y_pred.tail(n_steps),
        labels=["y_train", "y_test", "y_pred"],
    )
    plt.ylabel('Value')
    plt.title(j)
    plt.show()

# Forecasting with sktime
# Todo: compact the models in few lines
mapping = {
    ExponentialSmoothing: dict(
        trend=None, seasonal="add", sp=7 * 24
    ),  # None + add/additive (seasonal) gave us the best results
    make_reduction: dict(estimator = KNeighborsRegressor(n_neighbors=10),
    window_length=24 * 7, strategy="recursive"),
    # AutoETS: dict(auto=True, seasonal = "add", sp=7*24, n_jobs=-1,maxiter=10), #slow
    # BATS: dict(sp=7*24, use_trend=True, use_box_cox=False), #BATS: really slow
}
model_title = ["Exponential Smoothing", "KNeighborsRegressor"]#,"AutoETS","BATS"]

for (model_class, kwargs), title in zip(mapping.items(), model_title):
    model = model_class(**kwargs)
    model.fit(y=y_train, X=regressors_train)
    y_pred = model.predict(fh)
    plot_series(
        y_train.tail(n_steps),
        y_test.tail(n_steps),
        y_pred.tail(n_steps),
        labels=["y_train", "y_test", "y_pred"],
    )
    plt.title(title)
    plt.show()

########################################
# Todo: increase the noise and document somewhere what I'm doing
# The values used for the noise parameter are: 5,10,50,100
# The results are documented through the plots saved

# Todo: Investigate about the type of data for:
# time_of_day and day_of_week; if characters or integers

# time_of_day and day_of_week are two categorical variables since
# they can assume only one among a limited,
# and usually fixed, number of possible values.
# There are three common approaches for converting
# ordinal and categorical variables to numerical values:
# Ordinal Encoding ,One-Hot Encoding, Dummy Variable Encoding.
# I think in our case, One-Hot Encoding can be more appropriate tool.
# So, next thing to do.


# Todo: add models to dashboards
