# flake8: noqa
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

from pandas.tseries.frequencies import to_offset
import numpy as np
import pandas as pd
import timely_beliefs as tb


def filter_seasonalities(
    df: tb.BeliefsDataFrame,
    seasonalities: Dict[str, Union[int, List[int]]],
    n_steps: int,
    max_train_cut: float = 0.2,
) -> Tuple[Dict[str, List[int]], pd.Timestamp]:
    """Filter out seasonalities (periodic/seasonal regressors) that can't be applied
    without sacrificing too much training data, for a given n_steps to forecast past the last known event.

    Why does a seasonally lagged regressor sacrifice training data?
    Because too early a start time causes NaN values in the training data.
    To avoid that, we need to set the first applicable start time to the starting index + the seasonality

    When does a multi-step forecast causes seasonalities to be discarded?
    Any seasonalities that are lower than the forecast horizon itself will cause NaN values in the training data.
    While recurrent models may replace these NaN values with previous predictions, direct models will fail in this case.
    We currently discard any seasonalities lower than the forecast horizon.

    :param df: BeliefsDataFrame with historical observations only
    :param seasonalities: dictionary with the maximum number of periods per frequency string
                          for example:
                          - {"D": 7} denotes lagged regressors of 1 day up to 7 days
                          - {"D": 7, "Y": 2} denotes daily lags up to 7 days, and annual lags up to 2 years
                          Alternatively, specify the minimum number of periods, too, like:
                          - {"D": (3, 7)} denotes lagged regressors of 3 days up to 7 days
    :param n_steps: number of df.event_resolution steps into the future, for which observations are missing
    :param max_train_cut: maximum share of training data that can be cut for a given seasonality to remain applicable
    :returns: tuple comprising:
              - the seasonalities that can be applied without sacrificing too much training data
              - the first applicable event start to which all remaining seasonalities can apply
    """
    first_event = df.head(1)
    max_first_applicable_event_start = first_event.index.get_level_values(
        "event_start"
    )[0]
    seasonalities_left = seasonalities.copy()
    for freq, limit_periods in seasonalities.items():
        if isinstance(limit_periods, int):
            min_period = 1
            max_period = limit_periods
        else:
            limit_periods: Tuple[int, int]
            min_period = limit_periods[0]
            max_period = limit_periods[1]
        seasonalities_left[freq] = [min_period, max_period]
        for periods in range(min_period, max_period + 1):
            if periods * pd.Timedelta(freq) < pd.Timedelta(
                n_steps * df.event_resolution
            ):
                # Remove seasonalities that last shorter than the forecast horizon itself
                if periods == max_period:
                    # this seasonality is inapplicable
                    seasonalities_left.pop(freq)
                else:
                    # the applicable number of periods for this seasonality is limited
                    new_min_period = periods + 1
                    seasonalities_left[freq][0] = new_min_period
                # todo: continue?
            first_applicable_event_start = (
                first_event.index.get_level_values("event_start").shift(
                    periods=periods, freq=freq
                )
                # .shift(periods=n_steps, freq=df.event_resolution)
            )
            share_cut = len(
                df[
                    df.index.get_level_values("event_start")
                    < first_applicable_event_start[0]
                ]
            ) / len(df)
            if share_cut > max_train_cut:
                if periods == 1:
                    # this seasonality is inapplicable
                    seasonalities_left.pop(freq)
                else:
                    # the applicable number of periods for this seasonality is limited
                    new_max_period = periods - 1
                    seasonalities_left[freq][1] = new_max_period
                break
            max_first_applicable_event_start = max(
                max_first_applicable_event_start, first_applicable_event_start[0]
            )
    return seasonalities_left, max_first_applicable_event_start


def series_to_supervised(
    bdf: tb.BeliefsDataFrame,
    seasonalities: Dict[str, int],
    end_of_training: str,
    n_steps: int = 1,
    difference_seasonalities: Dict[str, int] = None,
    backcasting_allowed: bool = False,
    one_hot_encoding: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    {"1H": 24, "1D": 7, "Y": 2}
    Inputs:
    :param seasonalities: dictionary with the maximum number of periods per frequency string
                          for example:
                          - {"D": 7} denotes lagged regressors of 1 day up to 7 days
                          - {"D": 7, "Y": 2} denotes daily lags up to 7 days, and annual lags up to 2 years
                          Alternatively, specify the minimum number of periods, too, like:
                          - {"D": (3, 7)} denotes lagged regressors of 3 days up to 7 days
    :param n_steps: number of bdf.event_resolution steps into the future, for which observations are missing
    :param difference_seasonalities: same as seasonalities, but regressors represent the difference between
                                     the lagged variable and the most recent observation
    :param backcasting_allowed: if False, values after end_of_training will be explicitly discarded (no cheating),
                                if True, future values may be used as valuable insight for predicting missing values
                                # todo: consider adding {"H": 0} and using it to take into account previous predictions of the forecast event value
                                # todo: consider using {"H": (-1, 1)} for filling up a single missing value
    Outputs:
    :return train_dataset: dataframe contains train dataset with applicable regressors
    :return test_dataset: dataframe contains test dataset with applicable regressors
    """

    # Filter out seasonalities that cannot apply without sacrificing too much training data
    seasonalities_left, first_valid_event_start = filter_seasonalities(
        bdf, seasonalities, n_steps, max_train_cut=0.2
    )
    if difference_seasonalities is None:
        difference_seasonalities = {}
    difference_seasonalities_left, ds_first_valid_event_start = filter_seasonalities(
        bdf, difference_seasonalities, n_steps, max_train_cut=0.2
    )
    first_valid_event_start = max(
        first_valid_event_start, ds_first_valid_event_start
    )  # todo: better to filter seasonalities together
    freq_str = to_offset(bdf.event_resolution).freqstr
    df: pd.DataFrame = simplify_index(bdf)

    # Remove any future values after end_of_training (no cheating)
    if not backcasting_allowed:
        df = df.loc[:end_of_training]

    # Add empty rows for the test dataset of the forecasts to be
    last_start = df.event_starts[-1]
    n_steps_df = pd.DataFrame(
        [np.nan] * n_steps,
        index=pd.date_range(
            last_start,
            last_start + bdf.event_resolution * n_steps,
            freq=bdf.event_resolution,
            closed="right",
        ),
        columns=["event_value"],
    )
    df = pd.concat([df, n_steps_df])
    total_dataset = df.copy()

    # Construct regressor feature columns for timing with respect to day, week and year
    total_dataset["hour of the day"] = df.index.to_series().dt.hour.astype(int)
    total_dataset["day of the week"] = df.index.to_series().dt.dayofweek.astype(int)
    total_dataset["day of the year"] = df.index.to_series().dt.dayofyear.astype(int)

    # One-hot encode timing features
    if one_hot_encoding:
        # todo: implement one-hot encoding as a function applied on a specific column
        pass

    # Construct regressor feature columns for seasonalities
    for freq, limit_periods in seasonalities_left.items():
        min_period = limit_periods[0]
        max_periods = limit_periods[1]
        print(limit_periods)

        for periods in range(min_period, max_periods + 1):
            # if periods * pd.Timedelta(freq) < pd.Timedelta(time_horizon * resolution):
            #     # Ignore seasonalities that last shorter than the forecast horizon itself
            #     continue
            total_dataset[f"{periods}x{freq}"] = (
                df.shift(periods=periods, freq=freq)
                # .shift(periods=time_horizon - 1, freq=resolution)
            )

    # Construct regressor feature columns for difference seasonalities
    for freq, limit_periods in difference_seasonalities_left.items():
        min_period = limit_periods[0]
        max_periods = limit_periods[1]
        print(limit_periods)

        for periods in range(min_period, max_periods + 1):
            total_dataset[f"{periods}x{freq}-{n_steps}x{freq_str}"] = (
                df.shift(periods=periods, freq=freq)
                # .shift(periods=time_horizon - 1, freq=resolution)
                - df.shift(periods=n_steps, freq=freq_str)
            )

    # Split data into train and test periods
    train_dataset = total_dataset.loc[first_valid_event_start:end_of_training, :]
    test_dataset = total_dataset.loc[end_of_training + bdf.event_resolution :, :]
    #print("train and test datasets")
    #print(train_dataset)
    #print(test_dataset)
    return train_dataset, test_dataset


def forecasts_to_beliefs(
    forecasts: Union[np.ndarray, List[Union[float, int]]],
    sensor: tb.Sensor,
    forecaster: tb.BeliefSource,
    current_time: datetime,
) -> tb.BeliefsDataFrame:
    """Convert a list of forecast values into a BeliefsDataFrame.

    :param forecasts: list of forecast values
    :param sensor: timely-beliefs Sensor whose values the forecasts refer to
    :param forecaster: timely-beliefs Source that created the forecasts
    :param current_time: timezone-aware datetime at which the forecasts were made
    :returns: timely-beliefs BeliefsDataFrame describing forecasts as beliefs.
    """
    beliefs = [
        tb.TimedBelief(
            sensor=sensor,
            source=forecaster,
            event_start=current_time
            + sensor.event_resolution * h
            - sensor.event_resolution,
            belief_horizon=sensor.event_resolution * h,
            value=forecasts[h - 1],
        )
        for h in range(1, len(forecasts) + 1)
    ]
    return tb.BeliefsDataFrame(beliefs).convert_index_from_belief_time_to_horizon()


def simplify_index(
    bdf: tb.BeliefsDataFrame, index_levels_to_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Drops indices other than event_start.
    Optionally, salvage index levels as new columns.
    Because information stored in the index levels is potentially lost*,
    we cannot guarantee a complete description of beliefs in the BeliefsDataFrame.
    Therefore, we type the result as a regular pandas DataFrame.
    * The index levels are dropped (by overwriting the multi-level index with just the “event_start” index level).
      Only for the columns named in index_levels_to_columns, the relevant information is kept around.
    """
    if index_levels_to_columns is not None:
        for col in index_levels_to_columns:
            try:
                bdf[col] = bdf.index.get_level_values(col)
            except KeyError:
                if hasattr(bdf, col):
                    bdf[col] = getattr(bdf, col)
                else:
                    raise KeyError(f"Level {col} not found")
    bdf.index = bdf.index.get_level_values("event_start")
    return bdf


def prepare_df_for_sktime(bdf, current_time, n_events):
    current_bdf = bdf[bdf.index.get_level_values("event_start") < current_time]
    realised_bdf = bdf[
        (bdf.index.get_level_values("event_start") >= current_time)
        & (
            bdf.index.get_level_values("event_start")
            < current_time + timedelta(hours=n_events)
        )
    ]
    day_of_week = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    for i, x in enumerate(day_of_week):
        current_bdf[x] = (current_bdf.event_starts.weekday == i).astype(int)
        realised_bdf[x] = (realised_bdf.event_starts.weekday == i).astype(int)
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
        current_bdf[y] = (current_bdf.event_starts.hour == j).astype(int)
        realised_bdf[y] = (realised_bdf.event_starts.weekday == j).astype(int)

    n_steps = len(realised_bdf)
    # Define X and y (i.e. "regressors" and "y")
    regressors_train = current_bdf[day_of_week + time_of_day]
    regressors_test = realised_bdf[day_of_week + time_of_day]
    y_train = current_bdf["event_value"]
    y_test = realised_bdf["event_value"]
    # Fix Index
    regressors_train.index = regressors_train.index.get_level_values("event_start")
    regressors_train.index = regressors_train.index.to_period("H")
    y_train.index = y_train.index.get_level_values("event_start")
    y_train.index = y_train.index.to_period("H")
    y_test.index = y_test.index.get_level_values("event_start")
    y_test.index = y_test.index.to_period("H")
    return y_train, y_test, regressors_train, regressors_test
