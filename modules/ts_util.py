#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
#from statsmodels.tools.eval_measures import mse, rmse, meanabs
from sklearn.metrics import mean_squared_error, r2_score


# #### Augmented Dicky Fuller Test
# 1. If p-value>0.05: Weak evidence against the null hypothesis. Fail to reject the null hypothesis. Data has a unit root and is non-stationary.
# 2. If p-value<=0.05: Strong evidence against the null hypothesis. Reject the null hypothesis. Data has no unit root and is stationary.

#help(adfuller)
def get_adf_test_outputs(ts):
    test_outputs = adfuller(ts.dropna(), autolag='AIC')
    test_outputs_formatted = pd.Series(index=['ADF Test Statistic', 'p-value', '# Lags Used', '# Observations Used'], data=test_outputs[:4])
    critical_values = []
    for key,val in test_outputs[4].items():
        critical_values.append(pd.Series(index=['Critical Value ({})'.format(key)], data=val))

    return(test_outputs_formatted.append(critical_values))


# #### Granger Causality Tests

#help(grangercausalitytests)
def get_causality(ts1, ts2, maxlags):
    gr_caus = grangercausalitytests(ts1.join(ts2, how="inner", lsuffix='Conv', rsuffix='Org'), maxlags)
    
    return gr_caus


# #### Measure Prediction Errors:

def measure_pred_errors(actual, predicted):
    mse = mean_squared_error(actual, predicted).round(4)
    rmse = np.sqrt(mse).round(4)
    return mse,rmse

# #### Get "Integrated" or d value for ARIMA
# Number of times a series should be differenced till it is stationary

def find_d_value(ts, start=0, end=10):
    
    for d in range(start,end+1):
        if d>0:
            ts_diff = ts.diff(d)
            p_value = get_adf_test_outputs(ts_diff)[1]
        else:
            p_value = get_adf_test_outputs(ts)[1]
            
        if p_value<0.05:
            return d

### Get Rolling averages (mean/standard deviation) based on the window size        
def get_rolling_mean(ts, window):
    return ts.rolling(window).mean()

def get_rolling_std_dev(ts, window):
    return ts.rolling(window).std()

