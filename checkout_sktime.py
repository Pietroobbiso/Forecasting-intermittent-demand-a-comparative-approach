"""Go through the tutorial notebook with the water data + plots"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.bats import BATS
from sktime.forecasting.ets import AutoETS

import sktime_dl.classifiers.deeplearning as sdlcd

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction, TransformedTargetForecaster
from sklearn.neighbors import KNeighborsRegressor
from sktime.transformations.series.detrend import Deseasonalizer
from sklearn import tree
from sktime.utils.plotting import plot_series
from sktime.forecasting.theta import ThetaForecaster

from process_analytics.settings.settings import bdf, current_time, n_events
from process_analytics.utils.forecast_utils import prepare_df_for_sktime

pd.options.mode.chained_assignment = None
import time
startTime = time.time()

n_steps = 48
y_train, y_test, regressors_train, regressors_test = prepare_df_for_sktime(
    bdf, current_time, n_events
)

# Define fh
fh = ForecastingHorizon(y_test.index, is_relative=False)
# print(fh)
# Models:
#Overview of forecasting algorithms:
'''
['ARIMA',
 'AutoARIMA',
 'AutoETS',
 'BATS',
 'DirRecTabularRegressionForecaster',
 'DirRecTimeSeriesRegressionForecaster',
 'DirectTabularRegressionForecaster',
 'DirectTimeSeriesRegressionForecaster',
 'EnsembleForecaster',
 'ExponentialSmoothing',
 'ForecastingGridSearchCV',
 'ForecastingRandomizedSearchCV',
 'HCrystalBallForecaster',
 'MultioutputTabularRegressionForecaster',
 'MultioutputTimeSeriesRegressionForecaster',
 'MultiplexForecaster',
 'NaiveForecaster',
 'OnlineEnsembleForecaster',
 'PolynomialTrendForecaster',
 'Prophet',
 'RecursiveTabularRegressionForecaster',
 'RecursiveTimeSeriesRegressionForecaster',
 'StackingForecaster',
 'TBATS',
 'ThetaForecaster',
 'TransformedTargetForecaster']
'''
#Naïve forecasters (with and without seasonal periodicity)
# forecaster_title = ["NaiveForecaster", "NaiveForecaster with seasonal periodicity"]
# sp = [1, 2*7*24]
# for i, j in zip(sp, forecaster_title):
#     forecaster = NaiveForecaster(strategy="last", sp=i)
#     forecaster.fit(y=y_train, X=regressors_train)
#     y_pred = forecaster.predict(X=regressors_test,fh=fh)
#     # print(y_pred)
#     plot_series(
#         y_train.tail(n_steps),
#         y_test.tail(n_steps),
#         y_pred.tail(n_steps),
#         labels=["y_train", "y_test", "y_pred"],
#     )
#     plt.title(j)
#     plt.show()
#     print(mean_absolute_percentage_error(y_pred, y_test))

# Forecasting with sktime
mapping = {
    RecursiveTimeSeriesRegressionForecaster: dict(
            regressor=sdlcd.MLPRegressor(nb_epochs=500, verbose=False),
            window_length=1*24*7
    )
    #CNNRegressor(nb_epochs=400, kernel_size=7, verbose=False)
    # NaiveForecaster : dict(
    #     strategy="last", sp=7*24*1
    # #),
    # ExponentialSmoothing: dict(
    #     trend=None, seasonal="add", sp=1*7*24
    #), # None + add/additive (seasonal) gave us the best results
    #AutoARIMA:dict(sp=7*24*1,d=0, suppress_warnings=True),
        # make_reduction: dict(
        # estimator=KNeighborsRegressor(n_neighbors=10),
        # window_length=1*24*7,
        # strategy="multioutput",
    #),  # try also with "direct" and "multioutput"
    #ThetaForecaster: dict(sp=1*7*24),
    #AutoETS: dict(seasonal = "add", sp=7*24,maxiter=10), #slow
    #BATS: dict(sp=7*24, use_box_cox=False), #BATS: really slow
}
model_title = ['MLPRegressor']#["Naive Forecaster with Seasonality","Exponential Smoothing","KNeighborsRegressor","AutoETS"]#,"BATS"]#"Theta Forecaster"],#"ARIMA",]  # ,
for (model_class, kwargs), title in zip(mapping.items(), model_title):
    model = model_class(**kwargs)
    model.fit(y=y_train, X=regressors_train,fh=fh)
    y_pred = model.predict(X=regressors_test)
    #print(y_pred.values)
    plot_series(
        y_train.tail(n_steps),
        y_test.tail(n_steps),
        y_pred.tail(n_steps),
        labels=["y_train", "y_test", "y_pred"],
    )
    plt.title(title)
    plt.show()
    print(mean_absolute_percentage_error(y_pred, y_test))

# regressor = MLPRegressor(nb_epochs=500, verbose=False)
#
# forecaster = RecursiveTimeSeriesRegressionForecaster(
#                 regressor=regressor,
#                 window_length=window_length)
# forecaster.fit(train)
# predict = forecaster.predict(fh)


# forecaster = TransformedTargetForecaster(
#     [
#         ("deseasonalize", Deseasonalizer(model="additive", sp=1*7*24)),
#         (
#             "forecast",
#             make_reduction(
#                 ExponentialSmoothing(),
#                 scitype="tabular-regressor",
#                 strategy="recursive",
#             ),
#         ),
#     ]
# )
# forecaster.fit(y=y_train)#, X=regressors_train)
# y_pred = forecaster.predict(fh=fh)
# #print(y_pred.values)
# plot_series(
#     y_train.tail(n_steps),
#     y_test.tail(n_steps),
#     y_pred.tail(n_steps),
#     labels=["y_train", "y_test", "y_pred"],
# )
# plt.title("Exponential Smoothing")
# plt.show()
#print(mean_absolute_percentage_error(y_pred, y_test))
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
##Documented results: Some algorithms are defined so that they ignore exogenous variables (based on their typical mathematical model specification)
#1) Exponential Smoothing: with exogeneous variables: mape = 0.68 ; without exogenous variables: mape = 0.68 ( no difference); then when using 2*7*24 -> mape = 12.02; whne using 3*7*24 -> mape = 0.37
#2) Naive Forecaster with seasonal periodicity: with exogeneous variables: mape = 0.167; without exogenous variables: mape = 0.167 ( no difference); then when using 2*7*24 -> mape = 0.164; whne using 3*7*24 -> mape = 0.42
#3) ARIMA: without exogenous variables: mape = 0.
#4) Theta Forecaster: with exogeneous variables: mape = 0.266 ; without exogenous variables: mape = 0.266 ( no difference); when using 2*7*24 -> mape = 0.173; whne using 3*7*24 -> mape = 0.346
#5) The reduction approaches:
# 5.1) KNeighboursRegressor:
#  5.1.1): recursive approach: with exogeneous variables: mape = 0.18 ; without exogenous variables: mape = 0.0.18 ( no difference); when using 2*7*24 -> mape = 0.27; when using 3*7*24 -> mape = 0.27
#  5.1.2): direct approach: with exogeneous variables: mape = 0.17 ; without exogenous variables: mape = 0.17 ( no difference); when using 2*7*24 -> mape = 0.30; when using 3*7*24 -> mape = 0.19
#  5.1.3): multioutput approach: with exogeneous variables: mape = 0.16 ; without exogenous variables: mape = 0.17; when using 2*7*24 -> mape = 0.30; when using 3*7*24 -> mape = 0.19
# 5.2) SVR bad results (graphically), not comparable to the other models
# 5.3) Decision Tree Regressor