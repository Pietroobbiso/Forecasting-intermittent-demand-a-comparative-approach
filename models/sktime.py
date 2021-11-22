import pandas as pd
import timely_beliefs as tb

from sklearn.neighbors import KNeighborsRegressor


from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon

from sktime.forecasting.ets import AutoETS

from sktime.forecasting.exp_smoothing import ExponentialSmoothing

from sktime.forecasting.naive import NaiveForecaster

from sktime.forecasting.theta import ThetaForecaster

from process_analytics.utils import forecast_utils


def naive_forecaster(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    bdf: tb.BeliefsDataFrame,
    n_events: int,
) -> tb.BeliefsDataFrame:
    # todo: make sure that the forecaster does not receive bdf (it shouldn't need it)
    (
        y_train,
        y_test,
        regressors_train,
        regressors_test,
    ) = forecast_utils.prepare_df_for_sktime(bdf, current_time, n_events)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = tb.BeliefSource("Naive Forecaster")
    model = NaiveForecaster(strategy="last", sp=1)
    model.fit(y=y_train, X=regressors_train, fh=fh)
    y_pred = model.predict(X=regressors_test)
    # print(y_pred)

    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=y_pred.values,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    return forecast_bdf


def naive_forecaster_with_seasonality(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    bdf: tb.BeliefsDataFrame,
    n_events: int,
) -> tb.BeliefsDataFrame:
    # todo: make sure that the forecaster does not receive bdf (it shouldn't need it)
    (
        y_train,
        y_test,
        regressors_train,
        regressors_test,
    ) = forecast_utils.prepare_df_for_sktime(bdf, current_time, n_events)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = tb.BeliefSource("Naive Forecaster with seasonality")
    model = NaiveForecaster(strategy="last", sp=7 * 24)
    model.fit(y=y_train, X=regressors_train, fh=fh)
    y_pred = model.predict(X=regressors_test)
    # print(y_pred)

    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=y_pred.values,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    return forecast_bdf


def exponential_smoothing(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    bdf: tb.BeliefsDataFrame,
    n_events: int,
) -> tb.BeliefsDataFrame:
    # todo: make sure that the forecaster does not receive bdf (it shouldn't need it)
    (
        y_train,
        y_test,
        regressors_train,
        regressors_test,
    ) = forecast_utils.prepare_df_for_sktime(bdf, current_time, n_events)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = tb.BeliefSource("Exponential Smoothing")
    model = ExponentialSmoothing(trend=None, seasonal="add", sp=7 * 24)
    model.fit(y=y_train, X=regressors_train, fh=fh)
    y_pred = model.predict(X=regressors_test)


    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=y_pred.values,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    return forecast_bdf


def knearest_neighbors(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    bdf: tb.BeliefsDataFrame,
    n_events: int,
) -> tb.BeliefsDataFrame:
    # todo: make sure that the forecaster does not receive bdf (it shouldn't need it)
    (
        y_train,
        y_test,
        regressors_train,
        regressors_test,
    ) = forecast_utils.prepare_df_for_sktime(bdf, current_time, n_events)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = tb.BeliefSource("KNeighbor Regressor")
    model = make_reduction(
        estimator=KNeighborsRegressor(n_neighbors=10),
        window_length=24 * 7,
        strategy="recursive",
    )
    model.fit(y=y_train, X=regressors_train, fh=fh)
    y_pred = model.predict(X=regressors_test)
    # print(y_pred)

    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=y_pred.values,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    return forecast_bdf

def theta_forecaster(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    bdf: tb.BeliefsDataFrame,
    n_events: int,
) -> tb.BeliefsDataFrame:
    # todo: make sure that the forecaster does not receive bdf (it shouldn't need it)
    (
        y_train,
        y_test,
        regressors_train,
        regressors_test,
    ) = forecast_utils.prepare_df_for_sktime(bdf, current_time, n_events)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = tb.BeliefSource("Theta forecaster")
    model = ThetaForecaster(sp=7*24,deseasonalize = True)
    model.fit(y=y_train, X=regressors_train, fh=fh)
    y_pred = model.predict(X=regressors_test)
    # print(y_pred)

    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=y_pred.values,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    return forecast_bdf

def AutoETS_forecaster(
    current_bdf: tb.BeliefsDataFrame,
    current_time: pd.Timestamp,
    bdf: tb.BeliefsDataFrame,
    n_events: int,
) -> tb.BeliefsDataFrame:
    # todo: make sure that the forecaster does not receive bdf (it shouldn't need it)
    (
        y_train,
        y_test,
        regressors_train,
        regressors_test,
    ) = forecast_utils.prepare_df_for_sktime(bdf, current_time, n_events)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = tb.BeliefSource("AutoETS")
    model = AutoETS(seasonal="add",sp=7*24,maxiter=10)
    model.fit(y=y_train, X=regressors_train, fh=fh)
    y_pred = model.predict(X=regressors_test)
    # print(y_pred)

    forecast_bdf = forecast_utils.forecasts_to_beliefs(
        forecasts=y_pred.values,
        sensor=current_bdf.sensor,
        forecaster=forecaster,
        current_time=current_time,
    )
    return forecast_bdf



