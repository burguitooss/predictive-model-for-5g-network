# -*- coding: utf-8 -*-

################################### IMPORTS ###################################

from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import pandas as pd
from random import randint
import time


################################## FUNCTIONS ##################################

# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
    t,d,s,p,b,r = config
    # define model
    history = np.array(history)
    model = ExponentialSmoothing(history, trend=t, damped_trend=d,
                                 seasonal=s, seasonal_periods=p, use_boxcox=b)
    # fit model
    model_fit = model.fit(optimized=True, remove_bias=r)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]

# mean absolute scaled error or mase
def measure_mase(training, actual, predicted):
    n = training.shape[0]
    d = np.abs(  np.diff( training) ).sum()/(n-1)

    errors = np.abs(actual - predicted )
    return errors.mean()/d

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = exp_smoothing_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    mase_error = measure_mase(train, test, predictions)
    return mase_error

# score a model, return None on failure
def score_model(data, n_test, cfg):
    result = None
    # convert config to a key
    key = str(cfg)
    try:
        # never show warnings when grid searching, too noisy
        with catch_warnings():
            filterwarnings("ignore")
            result = walk_forward_validation(data, n_test, cfg)
    except:
        error = None
    return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
		# execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
    scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

# create a set of exponential smoothing configs to try
def exp_smoothing_configs(seasonal=[None]):
    models = list()
    # define config lists
    t_params = ['add', 'mul', None]
    d_params = [True, False]
    s_params = ['add', 'mul', None]
    p_params = seasonal
    b_params = [True, False]
    r_params = [True, False]
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t,d,s,p,b,r]
                            models.append(cfg)
    return models

# try different training/test set configurations
def check_best_training_test_division(test_series, test_percentaje, cfg_list):
    file = open('training_test_division_results.txt', 'w')
    for i in range(0, 3):
        file.write('------------------------------------- ')
        file.write(str(int(100-test_percentaje*100)) + '/' + 
                   str(int(test_percentaje*100)))
        file.write(' -------------------------------------\n')
        iteration = 1
        n_test = int(test_percentaje*data.shape[1])
        for i in test_series:
            score = grid_search(i, cfg_list, n_test)
            file.write('Series ' + str(iteration) + ' -> ')
            file.write('mase: ' + str(score[0][1]) + '\n')
            iteration = iteration + 1
        test_percentaje = test_percentaje - 0.1
        file.write('\n')
    file.close()

# execute the grid search to get and order the best results
def execute_grid_search(data, data_lower_range, data_upper_range, n_test,
                        cfg_list):
    best_results = [[], [], [], []]
    for i in range(data_lower_range, data_upper_range):
        score = grid_search(data[i], cfg_list, n_test)
        if score[0][0] in best_results[1]:
            index = best_results[1].index(score[0][0])
            best_results[0][index] = best_results[0][index] + 1
            if score[0][1] < best_results[2][index]:
                best_results[2][index] = score[0][1]
            if score[0][1] > best_results[3][index]:
                best_results[3][index] = score[0][1]
                
        else:
            best_results[0].append(1)
            best_results[1].append(score[0][0])
            best_results[2].append(score[0][1])
            best_results[3].append(score[0][1])
    return best_results

# export results to a csv
def export_results(best_results):
    results = {'occurrences': best_results[0], 'config': best_results[1],
                     'min_mase': best_results[2], 'max_mase': best_results[3]}
    results_df = pd.DataFrame(data=results)
    results_sorted_df = results_df.sort_values(by=['occurrences'],
                                               ascending=False)
    results_sorted_df.to_csv('hw_grid_search_results.csv', index=None)
    

#################################### MAIN ####################################

if __name__ == '__main__':
    start_time = time.time()
    # define dataset
    data = np.array(pd.read_parquet(
        'datasets\\hourlyTimeSeries_2months.parquet').values)
    # model configs
    cfg_list = exp_smoothing_configs(seasonal=[0, 24])
    
    # decide to try train/test configs or execute the grid search
    study_train_test_division = False
    if study_train_test_division:
        test_series = [data[randint(0,len(data))], data[randint(0,len(data))],
                       data[randint(0,len(data))], data[randint(0,len(data))],
                       data[randint(0,len(data))], data[randint(0,len(data))],
                       data[randint(0,len(data))], data[randint(0,len(data))],
                       data[randint(0,len(data))], data[randint(0,len(data))]]
        test_percentaje = 0.3
        check_best_training_test_division(test_series, test_percentaje, cfg_list)
    else:
        data_lower_range, data_upper_range = 0, 5
        n_test = int(0.1*data.shape[1])
    
        # grid search time series on the specified data range
        best_results = execute_grid_search(data, data_lower_range,
                                           data_upper_range, n_test, cfg_list)
        export_results(best_results)
    
    end_time = time.time()
    print('Elapsed time: ' + str(end_time-start_time))
