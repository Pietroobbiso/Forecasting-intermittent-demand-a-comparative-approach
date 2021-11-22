from typing import Dict

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR

import pandas as pd

# import numpy as np
import copy
import timely_beliefs as tb

from process_analytics.utils import forecast_utils

# seasonality corresponding to current_bdf.event_resolution should be
# higher than n_events in order to take into account the most recent
# observation for every forecast horizon
DEFAULT_SEASONALITIES = {"1H": 4, "1D": 3, "7D": 1}
DEFAULT_DIFFERENCE_SEASONALITIES = {"1H": 4, "1D": 3, "7D": 1}


def linear_regression_forecaster(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    n_events: int,
    seasonalities: Dict[str, int] = {},
    difference_seasonalities: Dict[str, int] = None,
) -> tb.BeliefsDataFrame:
    """Forecaster: linear regression."""
    forecaster = tb.BeliefSource("Linear regression forecaster")

    if seasonalities is None:
        seasonalities = DEFAULT_SEASONALITIES
    if difference_seasonalities is None:
        difference_seasonalities = DEFAULT_DIFFERENCE_SEASONALITIES
    # Direct model for each step
    y_preds = []
    hybrid_bdf = copy.copy(current_bdf)
    #print(f"current_bdf = {current_bdf}")
    for time_horizon in range(1, n_events + 1):
        train_dataset, test_dataset = forecast_utils.series_to_supervised(
            hybrid_bdf,
            seasonalities=seasonalities,
            difference_seasonalities=difference_seasonalities,
            n_steps=time_horizon,
            end_of_training=current_time - current_bdf.event_resolution,
        )
        X_train = train_dataset.loc[:, train_dataset.columns != "event_value"]
        y_train = train_dataset["event_value"]
        X_test = test_dataset.loc[:, test_dataset.columns != "event_value"]
        # y_test = test_dataset[
        #     "event_value"
        # ]  # todo: check all NaN values and not used to fit or predict
        #print(f"train_dataset = {train_dataset}")
        #print(f"X_train = {X_train}")
        #print(f"X_tests = {X_test}")
        #print(f"current_bdf = {current_bdf}")
        model = LinearRegression()
        model.fit(X_train, y_train)
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_test.values)
        y_preds.append(y_pred[0])
        temp_forecast_bdf = forecast_utils.forecasts_to_beliefs(
            forecasts=y_preds,
            sensor=current_bdf.sensor,
            forecaster=forecaster,
            current_time=current_time,
        )
        #print(f"temp_forecast = {temp_forecast_bdf}")

        #print(f"y_pred = {y_pred}")

        #print(f"current_bdf = {current_bdf}")
        hybrid_bdf = pd.concat([current_bdf, temp_forecast_bdf]).sort_index()
        #print(f"hybrid_bdf = {hybrid_bdf}")

    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=y_preds,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    #print(forecast_bdf)
    return forecast_bdf


def random_forest_forecaster(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    n_events: int,
    seasonalities: Dict[str, int] = {},
    difference_seasonalities: Dict[str, int] = None,
) -> tb.BeliefsDataFrame:
    """Forecaster: linear regression."""
    forecaster = tb.BeliefSource("Random forest forecaster")

    if seasonalities is None:
        seasonalities = DEFAULT_SEASONALITIES
    if difference_seasonalities is None:
        difference_seasonalities = DEFAULT_DIFFERENCE_SEASONALITIES
    # Direct model for each step
    y_preds = []
    hybrid_bdf = copy.copy(current_bdf)
    #print(f"current_bdf = {current_bdf}")
    for time_horizon in range(1, n_events + 1):
        train_dataset, test_dataset = forecast_utils.series_to_supervised(
            hybrid_bdf,
            seasonalities=seasonalities,
            difference_seasonalities=difference_seasonalities,
            n_steps=time_horizon,
            end_of_training=current_time - current_bdf.event_resolution,
        )
        X_train = train_dataset.loc[:, train_dataset.columns != "event_value"]
        y_train = train_dataset["event_value"]
        X_test = test_dataset.loc[:, test_dataset.columns != "event_value"]
        # y_test = test_dataset[
        #     "event_value"
        # ]  # todo: check all NaN values and not used to fit or predict
        #print(f"train_dataset = {train_dataset}")
        #print(f"X_train = {X_train}")
        #print(f"X_tests = {X_test}")
        #print(f"current_bdf = {current_bdf}")
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_test.values)
        y_preds.append(y_pred[0])
        temp_forecast_bdf = forecast_utils.forecasts_to_beliefs(
            forecasts=y_preds,
            sensor=current_bdf.sensor,
            forecaster=forecaster,
            current_time=current_time,
        )
        #print(f"temp_forecast = {temp_forecast_bdf}")

        #print(f"y_pred = {y_pred}")

        #print(f"current_bdf = {current_bdf}")
        hybrid_bdf = pd.concat([current_bdf, temp_forecast_bdf]).sort_index()
        #print(f"hybrid_bdf = {hybrid_bdf}")

    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=y_preds,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    return forecast_bdf


def decision_tree_forecaster(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    n_events: int,
    seasonalities: Dict[str, int] = {},
    difference_seasonalities: Dict[str, int] = None,
) -> tb.BeliefsDataFrame:
    """Forecaster: linear regression."""
    forecaster = tb.BeliefSource("Decision Tree forecaster")

    if seasonalities is None:
        seasonalities = DEFAULT_SEASONALITIES
    if difference_seasonalities is None:
        difference_seasonalities = DEFAULT_DIFFERENCE_SEASONALITIES

    # Direct model for each step
    y_preds = []
    hybrid_bdf = copy.copy(current_bdf)
    #print(f"current_bdf = {current_bdf}")
    for time_horizon in range(1, n_events + 1):
        train_dataset, test_dataset = forecast_utils.series_to_supervised(
            hybrid_bdf,
            seasonalities=seasonalities,
            difference_seasonalities=difference_seasonalities,
            n_steps=time_horizon,
            end_of_training=current_time - current_bdf.event_resolution,
        )
        X_train = train_dataset.loc[:, train_dataset.columns != "event_value"]
        y_train = train_dataset["event_value"]
        X_test = test_dataset.loc[:, test_dataset.columns != "event_value"]
        # y_test = test_dataset[
        #     "event_value"
        # ]  # todo: check all NaN values and not used to fit or predict
        #print(f"train_dataset = {train_dataset}")
        #print(f"X_train = {X_train}")
        #print(f"X_tests = {X_test}")
       #print(f"current_bdf = {current_bdf}")
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_test.values)
        y_preds.append(y_pred[0])
        temp_forecast_bdf = forecast_utils.forecasts_to_beliefs(
            forecasts=y_preds,
            sensor=current_bdf.sensor,
            forecaster=forecaster,
            current_time=current_time,
        )
        #print(f"temp_forecast = {temp_forecast_bdf}")

        #print(f"y_pred = {y_pred}")

        #print(f"current_bdf = {current_bdf}")
        hybrid_bdf = pd.concat([current_bdf, temp_forecast_bdf]).sort_index()
        #print(f"hybrid_bdf = {hybrid_bdf}")

    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=y_preds,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    return forecast_bdf


def k_nearest_neighbours_forecaster(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    n_events: int,
    seasonalities: Dict[str, int] = {},
    difference_seasonalities: Dict[str, int] = None,
) -> tb.BeliefsDataFrame:
    """Forecaster: linear regression."""
    forecaster = tb.BeliefSource("K nearest neighbours forecaster")

    if seasonalities is None:
        seasonalities = DEFAULT_SEASONALITIES
    if difference_seasonalities is None:
        difference_seasonalities = DEFAULT_DIFFERENCE_SEASONALITIES

    # Direct model for each step
    y_preds = []
    hybrid_bdf = copy.copy(current_bdf)
    print(f"current_bdf = {current_bdf}")
    for time_horizon in range(1, n_events + 1):
        train_dataset, test_dataset = forecast_utils.series_to_supervised(
            hybrid_bdf,
            seasonalities=seasonalities,
            difference_seasonalities=difference_seasonalities,
            n_steps=time_horizon,
            end_of_training=current_time - current_bdf.event_resolution,
        )
        X_train = train_dataset.loc[:, train_dataset.columns != "event_value"]
        y_train = train_dataset["event_value"]
        X_test = test_dataset.loc[:, test_dataset.columns != "event_value"]
        # y_test = test_dataset[
        #     "event_value"
        # ]  # todo: check all NaN values and not used to fit or predict
        print(f"train_dataset = {train_dataset}")
        print(f"X_train = {X_train}")
        print(f"X_tests = {X_test}")
        print(f"current_bdf = {current_bdf}")
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_test.values)
        y_preds.append(y_pred[0])
        temp_forecast_bdf = forecast_utils.forecasts_to_beliefs(
            forecasts=y_preds,
            sensor=current_bdf.sensor,
            forecaster=forecaster,
            current_time=current_time,
        )
        print(f"temp_forecast = {temp_forecast_bdf}")

        print(f"y_pred = {y_pred}")

        print(f"current_bdf = {current_bdf}")
        hybrid_bdf = pd.concat([current_bdf, temp_forecast_bdf]).sort_index()
        print(f"hybrid_bdf = {hybrid_bdf}")

    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=y_preds,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    return forecast_bdf


def support_vector_machine_forecaster(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    n_events: int,
    seasonalities: Dict[str, int] = {},
    difference_seasonalities: Dict[str, int] = None,
) -> tb.BeliefsDataFrame:
    """Forecaster: linear regression."""
    forecaster = tb.BeliefSource("Support Vector Machine forecaster")

    if seasonalities is None:
        seasonalities = DEFAULT_SEASONALITIES
    if difference_seasonalities is None:
        difference_seasonalities = DEFAULT_DIFFERENCE_SEASONALITIES

    # Direct model for each step
    y_preds = []
    hybrid_bdf = copy.copy(current_bdf)
    #print(f"current_bdf = {current_bdf}")
    for time_horizon in range(1, n_events + 1):
        train_dataset, test_dataset = forecast_utils.series_to_supervised(
            hybrid_bdf,
            seasonalities=seasonalities,
            difference_seasonalities=difference_seasonalities,
            n_steps=time_horizon,
            end_of_training=current_time - current_bdf.event_resolution,
        )
        X_train = train_dataset.loc[:, train_dataset.columns != "event_value"]
        y_train = train_dataset["event_value"]
        X_test = test_dataset.loc[:, test_dataset.columns != "event_value"]
        # y_test = test_dataset[
        #     "event_value"
        # ]  # todo: check all NaN values and not used to fit or predict
        #print(f"train_dataset = {train_dataset}")
        #print(f"X_train = {X_train}")
        #print(f"X_tests = {X_test}")
        #print(f"current_bdf = {current_bdf}")
        model = SVR()
        model.fit(X_train, y_train)
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_test.values)
        y_preds.append(y_pred[0])
        temp_forecast_bdf = forecast_utils.forecasts_to_beliefs(
            forecasts=y_preds,
            sensor=current_bdf.sensor,
            forecaster=forecaster,
            current_time=current_time,
        )
        #print(f"temp_forecast = {temp_forecast_bdf}")

        #print(f"y_pred = {y_pred}")

        #print(f"current_bdf = {current_bdf}")
        hybrid_bdf = pd.concat([current_bdf, temp_forecast_bdf]).sort_index()
        #print(f"hybrid_bdf = {hybrid_bdf}")

    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=y_preds,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    return forecast_bdf


def arima_forecaster(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    n_events: int,
    seasonalities: Dict[str, int] = None,
    difference_seasonalities: Dict[str, int] = None,
) -> tb.BeliefsDataFrame:
    """Forecaster: linear regression."""
    forecaster = tb.BeliefSource("ARIMA forecaster")

    if seasonalities is None:
        seasonalities = DEFAULT_SEASONALITIES
    if difference_seasonalities is None:
        difference_seasonalities = DEFAULT_DIFFERENCE_SEASONALITIES

    # Direct model for each step
    # y_preds = []
    # for time_horizon in range(1, n_events + 1):
    train_dataset, test_dataset = forecast_utils.series_to_supervised(
        current_bdf,
        seasonalities=seasonalities,
        difference_seasonalities=difference_seasonalities,
        n_steps=48,
        end_of_training=current_time - current_bdf.event_resolution,
    )
    # X_train = train_dataset.loc[:, train_dataset.columns != "event_value"]
    # y_train = train_dataset["event_value"]
    # X_test = test_dataset.loc[:, test_dataset.columns != "event_value"]
    # y_test = test_dataset[
    #     "event_value"
    # ]  # todo: check all NaN values and not used to fit or predict
    # X_train = X_train.dropna()
    # y_train = y_train.dropna()
    # X_test = X_test.dropna()
    # print(f"X_train = {X_train}")
    # print(f"y_train = {y_train}")
    # print(f"X_test = {X_test}")
    history = [x for x in train_dataset]
    predictions = list()
    for t in range(len(test_dataset)):
        model = ARIMA(history, order=(48, 0, 1))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_dataset[t]
        history.append(obs)
    # model = ARIMA(current_bdf.values, order=(5,1,0))
    # model.fit()
    # model.fit(X_train.values, y_train.values)
    # y_pred = model.predict(X_test.values)
    # y_preds.append(y_pred[0])

    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=history,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    return forecast_bdf
