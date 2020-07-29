#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Python libraries
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime
from math import sqrt

#import lightgbm as lgbm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from scipy.stats import randint as sp_randint
from scipy.stats import uniform 
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanAbsolutePercentageError, MeanAbsoluteError
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.models import load_model

import warnings
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

from tqdm import tqdm

warnings.filterwarnings('ignore')

from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# In[2]:


# Define sub-functions as internal protected functions


# In[3]:


def ts_to_lagged_df(train_data, lag = 12, dropna = True):
    """
    
    """
    train_data_lagged = train_data.copy()
    for i in range(1, lag + 1):
        train_data_lagged["lag_" + str(i)] = train_data.iloc[:, 0].shift(i)
    if dropna:
        train_data_lagged.dropna(inplace = True)
    return train_data_lagged


# In[4]:


def train_test_split(data, n_test): 
    """
    """
    return data[:-n_test], data[-n_test:]


# In[5]:


mape = MeanAbsolutePercentageError()


# In[6]:


mae = MeanAbsoluteError()


# In[7]:


def get_naive_predictions(data, numPointsPredict, period):
    """
    Creates full_period_predictions which serves as a model for the naive modeling
    Returns numPointsPredict of predictions of the naive model from looping over the full period of predictions
    """
    ts=data[data.columns[0]]
    full_period_predictions=pd.DataFrame(columns=['date','data'])
    full_period_predictions["data"]=[0]*period

    for i in range(0, period):
        sum_data, count_dates,lastDate=0,0,None
        #increment by period to get that date every year
        #keep track of the number of dates because finding average and the last date for that spot in the period
        for j in range(i, len(ts), period):
            sum_data+=ts[j]
            count_dates+=1
            lastDate=data.index[j]
        full_period_predictions["date"][i]=pd.Timestamp.date(lastDate)
        full_period_predictions["data"][i]=sum_data/count_dates

    lastDate=full_period_predictions['date'][0]
    lastDateIndex=0
    #find the most recent date to start at
    for i in range(0,len(full_period_predictions['data'])):
        if full_period_predictions["date"][i]>lastDate:
            lastDate=full_period_predictions["date"][i]
            lastDateIndex=i

    predictions=pd.DataFrame(columns=['data'])
    predictions["data"]=[0]*numPointsPredict

    #start predicting from the last point +1 onwards
    for i in range(0,numPointsPredict):
        lastDateIndex+=1
        #if you reach the end of the period, reset to beginning of full_period_predictions
        if lastDateIndex>=period:
            lastDateIndex=0
        predictions['data'][i]=full_period_predictions["data"][lastDateIndex]
        
    return predictions


# In[8]:


def predict_with_insample_naive(train_data, forecast_len, period = 12):
  pred_naive = get_naive_predictions(train_data, forecast_len, period).values
  return pred_naive


# In[9]:


def get_scaled_Xy(data, test_from = '2019-01', lag = 12):
    """
    get scaled X and y for LSTM fitting

    Parameters
    ----------
    data : DataFrame
        DataFrame that restores time series

    Returns
    -------
    scaler : MinMaxScaler
        Scaler object that is used to convert data into [0, 1] scaled and will be used when transforming
    X_train: ndarray
        An array object that includes features, which is lagged time series at t-1 ... t-n (scaled)
    y_train: ndarry
        An array object that includes output, which is time series data at time t (scaled)
    X_test: ndarray
        An array object that includes features, which is lagged time series at t-1 ... t-n (scaled)
    y_test: ndarry
        An array object that includes output, which is time series data at time t (scaled)
    """
    scaler = MinMaxScaler()
    scaler.fit(data.values)
    scaled = scaler.transform(data.values)
    scaled_data = data.copy()
    scaled_data.iloc[:, 0] = scaled
    
    lag_scaled_data = ts_to_lagged_df(scaled_data, lag = lag)
    lag_scaled_data_train, lag_scaled_data_test = lag_scaled_data.loc[lag_scaled_data.index < test_from, :], lag_scaled_data.loc[lag_scaled_data.index >= test_from, :],
  
    X_train, y_train = lag_scaled_data_train.iloc[:, 1:].values, lag_scaled_data_train.iloc[:, 0].values
    X_train = np.reshape(X_train, X_train.shape + (1,))
    X_test, y_test = lag_scaled_data_test.iloc[:, 1:].values, lag_scaled_data_test.iloc[:, 0].values
    X_test = np.reshape(X_test, X_test.shape + (1,))
    
    return scaler, X_train, y_train, X_test, y_test


# In[10]:


def plot_train_process(fitted_model, metric = 'loss'):
  losses = fitted_model.history.history[metric]
  val_losses = fitted_model.history.history['val_' + metric]
  plt.figure(figsize=(12,4))
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.xticks(np.arange(0,len(losses),1))
  plt.plot(range(len(losses)),losses, label = metric);
  plt.plot(range(len(val_losses)),val_losses, label = "val_" + metric);
  plt.legend()
  plt.show()


# In[11]:


def forward_predict(model, scaler, X_test, y_test, lag = 12):
    """
    returns predictive values. NN-based model can predict only one-time step. e.g. it can forcasting a value at t + 1.
    So, iteratively, create predictive values for n-time steps. n means the length(time steps) that is the same as len(test).

    Parameters
    ----------
    data : DataFrame
        DataFrame that restores time series
    scaler : MinMaxScaler
        Scaler object that is used to convert data into [0, 1] scaled and is used when transforming
        I assume the scaler is the return argument from get_scaled_Xy().
    X_test: ndarray
        An array object that includes features, which is lagged time series at t-1 ... t-n
        I assume the scaler is the return argument from get_scaled_Xy().
    y_test: ndarry
        An array object that includes output, which is time series data at time t
        I assume the scaler is the return argument from get_scaled_Xy().
    Returns
    -------
    pred : DataFrame
        Time series data of predictive values. 
    """
    lstm_predictions_scaled = list()

    batch = X_test[0]
    current_batch = batch.reshape((1, lag, 1))

    for i in range(len(y_test)):  
        lstm_pred = model.predict(current_batch)[0]
        lstm_predictions_scaled.append(lstm_pred) 
        current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
        
    pred = scaler.inverse_transform(lstm_predictions_scaled)
    #pred = pd.DataFrame(pred, index = test_data.index, columns = test_data.columns)
    pred
    
    true = scaler.inverse_transform(y_test.reshape(len(y_test), 1))
    
    return pred, true


# In[12]:


def evaluate_perfomance(pred, true, plot = True, metric = "mape", pred_naive = None):
    is_mape = metric == "mape"
    is_mase = (metric == "mase") and (pred_naive is not None)

    if is_mape:
      ename = "Mean Absolute Percentage Error"
      error = mape(pred, true).numpy()
    elif is_mase:
      ename = "Mean Absolute Scaled Error"
      error =  mae(pred, true).numpy() / mae(pred_naive, true).numpy()

    if plot:
        # Plot the estimated values  実データと予測結果の図示
        plt.plot(true, label="test")
        plt.plot(pred, "r", label="pred")
        if is_mase:
          plt.plot(pred_naive, "g", label="pred_naive")
        plt.title(f"{ename}: {error:.2f}")
        plt.legend()
        plt.show()
    return error


# In[13]:


def build_CNN_base(n_input = 12, n_filters = 512, n_kernel = 3, optimizer = 'Adam', l2_penalty = 0.01, learning_rate=0.001,):
    
    n_input = n_input
    n_features= 1
    
    # define model
    model = Sequential()
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu', kernel_regularizer = l2(l2_penalty), input_shape=(n_input, 1)))
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu', kernel_regularizer = l2(l2_penalty))) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mape'])

    return model


# In[14]:


def get_opt_params_by_randomized_search_cv(X_train, y_train):
    # pass in fixed parameters n_input and n_class
    model_keras = KerasRegressor(
        build_fn = build_CNN_base,
        epochs=150, 
        batch_size=12, 
        #callbacks = callbacks,
        verbose=0
    )
    
    # parameter ranges to use in randomized search
    n_filters_opts  = [1, 2, 4, 8, 32, 64]
    n_kernel_opts = list(range(2, 12))
    l2_penalty_opts = [0.0, 0.01, 0.25, 0.50]

    keras_param_options = {
        'n_filters': n_filters_opts,
        'n_kernel': n_kernel_opts,
        'l2_penalty': l2_penalty_opts
    }
    
    # Create a RandomizedSearchCV instance, and then fit.
    rs_keras = RandomizedSearchCV( 
        model_keras, 
        param_distributions = keras_param_options,
        scoring = 'neg_mean_absolute_error',
        n_iter = 10, 
        cv = 5,
        n_jobs = -1,
        verbose = 1
    )
    rs_keras.fit(X_train, y_train)

    print('Best score obtained: {0}'.format(rs_keras.best_score_))
    print('Parameters:')
    for param, value in rs_keras.best_params_.items():
        print('\t{}: {}'.format(param, value))
        
    return rs_keras.best_params_


# In[15]:


def augment_data(X_train, y_train, times = 1000):
  original_X = X_train
  original_y = y_train
  for i in range(times - 1):
    augmented_X = original_X + np.random.normal(loc=0, scale=0.1, size=original_X.shape)
    X_train = np.append(X_train, augmented_X, axis = 0,)
    augmented_y = original_y + np.random.normal(loc=0, scale=0.1, size=original_y.shape)
    y_train = np.append(y_train, augmented_y, axis = 0,)

  return X_train, y_train


# In[16]:


def calculate_errors_with_CNN(data, plot = False, randomized_search = False, augmentation = False, augment_mag = 1000, verbose = True):
    
    pred, emase, emape = None, np.inf, np.inf
    
    if type(data) == pd.Series:
        data = data.to_frame()
    
    # remove the period affected by epidemic of corona virus
    building = data
    index_two_thirds=int((len(building) / 3) * 2 / 12) * 12
    start_date=pd.Timestamp.date(building.index[index_two_thirds])
    end_date=pd.Timestamp.date(building.index[-1])
    
    
    # checking whether  data of a building has enough number of its observations, 36 weeks, to evaluate the model performance
    if not building.isnull().values.any():
        # Data preparation
        scaler, X_train, y_train, X_test, y_test = get_scaled_Xy(building, test_from=building.index[index_two_thirds])
        
        # Data augmentation
        batch_size = 12

        if augmentation:
            X_train, y_train = augment_data(X_train, y_train, times = augment_mag)
            batch_size = max(1, int(X_train.shape[0] / 100))
            if verbose: print(f"Data augmentation: #training observations = {X_train.shape[0]}, batch_size = {batch_size}")

        # Create model with default params
        # the default params are examined in advance. See Neural_Network_Based_Methods_with_monthly_data.ipynb
        model = build_CNN_base(n_kernel = 5, n_filters = 64, l2_penalty = 0.25)
        
        if randomized_search:
            opt_params = get_opt_params_by_randomized_search_cv(X_train, y_train)
            if verbose: print(f"Randomized search: #opt_params = {opt_params}")
            model = build_CNN_base(**opt_params)
        
        
        # early stopping with patience
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=20)
        # model checkpoint to save the best model
        mc = ModelCheckpoint('best_model.h5', monitor='val_mae', mode='min', verbose=verbose, save_best_only=True)
        model.fit(X_train, y_train, batch_size = batch_size, epochs = 150, verbose = verbose, validation_data = (X_test, y_test), callbacks=[es, mc])

        #plot_train_process(model)
        best_model = load_model("best_model.h5")
        # Forcast with the test dataset
        pred, true = forward_predict(best_model, scaler, X_test, y_test)
        # Get MAPE
        pred_naive = predict_with_insample_naive(building.iloc[0:index_two_thirds, :], len(y_test))
        emape = evaluate_perfomance(pred, true, plot = plot, metric="mape")
        emase = evaluate_perfomance(pred, true, plot = plot, metric="mase", pred_naive=pred_naive)
        # set the accuracy into result Data Frame
        if verbose: print(f"MASE = {emase} MAPE = {emape}%")
        
    else:
        print(f"skip evaluating: {col}")
     
    return emase, emape, start_date, end_date


# In[17]:


def predict_with_CNN(data, forecast_len, plot = False, randomized_search = False, augmentation = False, augment_mag = 1000, verbose = True):
    
    pred, emase, emape, start_date, end_date = None, *calculate_errors_with_CNN(data, 
                                                        plot = plot, 
                                                        randomized_search=randomized_search, 
                                                        augmentation=augmentation, 
                                                        augment_mag=augment_mag, 
                                                        verbose=verbose)
    
    if type(data) == pd.Series:
        data = data.to_frame()
    
    # use all observations in the input data
    building = data
    
    # checking whether  data of a building has enough number of its observations, 36 weeks, to evaluate the model performance
    if not building.isnull().values.any():
        # Data preparation
        # To use whole data, I create X_test and y_test whose lenghts are zero.
        # So, I set the maximum value of datetime64[ns], 2262-04
        scaler, X_train, y_train, X_test, y_test = get_scaled_Xy(data, test_from = '2262-01')
        
        # Instead, I create dummy X_test and y_test for forward_predict function.
        # X_test[0] and the lenght of y_test is used in forward_prediction function.
        X_test = np.append(X_train[-1, 1:], y_train[-1]).reshape((1, 12, 1))
        y_test = np.array(range(forecast_len))
        
        # Data augmentation
        batch_size = 12

        if augmentation:
            X_train, y_train = augment_data(X_train, y_train, times = augment_mag)
            batch_size = max(1, int(X_train.shape[0] / 100))
            if verbose: print(f"Data augmentation: #training observations = {X_train.shape[0]}, batch_size = {batch_size}")

        # Create model with default params
        # the default params are examined in advance. See Neural_Network_Based_Methods_with_monthly_data.ipynb
        model = build_CNN_base(n_kernel = 5, n_filters = 64, l2_penalty = 0.25)
        
        if randomized_search:
            opt_params = get_opt_params_by_randomized_search_cv(X_train, y_train)
            if verbose: print(f"Randomized search: #opt_params = {opt_params}")
            model = build_CNN_base(**opt_params)
        
        
        # early stopping with patience
        # This time, there is no validation(test) data because I use whole data as the training data.
        # Hence, I observe only loss.
        es = EarlyStopping(monitor='loss', mode='min', verbose=verbose, patience=20)
        # model checkpoint to save the best model
        mc = ModelCheckpoint('best_model.h5', monitor='mae', mode='min', verbose=verbose, save_best_only=True)
        model.fit(X_train, y_train, batch_size = batch_size, epochs = 150, verbose = verbose, callbacks=[es, mc])

        best_model = load_model("best_model.h5")
        # Forcast with the test dataset
        pred, _ = forward_predict(best_model, scaler, X_test, y_test)
        
    else:
        print(f"skip evaluating: {col}")
     
    return pred, emase, emape, start_date, end_date


# In[18]:


# # Debugging
# !ls ../data/formatted_3year/cleaned/monthly
# path = "../data/formatted_3year/cleaned/monthly/"


# In[19]:


# chw = pd.read_csv(path + "cleaned_chw_monthly_01-01-2017_06-14-2020.csv", na_values=" ")
# chw.set_index(pd.DatetimeIndex(chw['dt']), inplace = True)


# In[20]:


# data = chw.loc[:, ["[7547] Duke Hospital - Ancillary Bldg, Hospital"]]


# In[21]:


# forecast_len = 24
# pred, emase, emape = predict_with_CNN(data, forecast_len, plot = False, randomized_search = False, augmentation = False, augment_mag = 1000, verbose = False)


# In[22]:


# print(f"{pred} {emase} {emape}")


# In[23]:


# pred_naive = get_naive_predictions(data, forecast_len, 12)


# In[24]:


# pred_df = pd.DataFrame(pred_naive)
# pred_df.index = pd.date_range(start=data.index[-1] , periods= forecast_len + 1, freq = 'M')[1:]


# In[25]:


# plt.plot(data, label="y_train")
# plt.plot(pred_df, "r", label="pred")
# #plt.title(f"{ename}: {error:.2f}")
# plt.legend()
# plt.show()


# In[26]:


# pred_naive = predict_with_insample_naive(train)


# In[27]:


# evaluate_perfomance(pred)


# In[28]:


# index_two_thirds=int((len(data)/3)*2)


# In[29]:


# calculate_errors_with_CNN(data, plot = True, randomized_search = False, augmentation = False, augment_mag = 1000, verbose = True)


# In[30]:


#pred, emase, emape, start_date, end_date = predict_with_CNN(data, forecast_len, plot = False, randomized_search = False, augmentation = False, augment_mag = 1000, verbose = True)


# In[31]:


#print(pred, emase, emape, start_date, end_date)


# In[ ]:




