#import statements
import pandas as pd
import numpy as np
from datetime import datetime
import statistics
import statsmodels.tsa.stattools
import statsmodels.tsa.statespace
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import math
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.simplefilter('ignore')


def forecast(data, period, freq, numPointsPredict):
    """
    Given data should be indexed dataframe with single column that contains utility data
    Performs the holt winters forecasting on the given data. Also needs the period and frequency
    of the data as well as the number of points to predict
    Returns the predictions for the data and the mean absolute percent error, mean absolute scaled error, 
    and the start and end dates for the test data
    Given more than 2 full periods of data
    """
    col=data.columns[0] #ignore the rest of the columns if more than one
    ts = data[col]

    train_data, test_data=get_train_test_data(data, period, freq)

    #initalize the model- one with damped true and one with damped as false
    holtWinters_damped_true = ExponentialSmoothing(endog=train_data,seasonal_periods=period,trend='add',seasonal='add',\
                                       damped=True)
    holtWinters_damped_false = ExponentialSmoothing(endog=train_data,seasonal_periods=period,trend='add',seasonal='add',\
                                       damped=False)

    #fit the model, chose not to specify any parameters
    holtWinters_damped_true=holtWinters_damped_true.fit()
    holtWinters_damped_false=holtWinters_damped_false.fit()
    
    #determine which model fits better (damped true or damped false) 
    #determine_lower_mae returns that true if damped true better and false is damped false better
    #create a new model using all the data, not just train data
    damped=determine_lower_mae(holtWinters_damped_true, holtWinters_damped_false, test_data)
    holtModelToUse = ExponentialSmoothing(endog=data,seasonal_periods=period,trend='add',seasonal='add',\
                                       damped=damped).fit()
    
    #get the predictions for the model
    predictions=holtModelToUse.predict((len(ts)-1), (len(ts)-1+numPointsPredict))
    
    #if change predictions are negative, make them 0
    predictions[predictions<0]=0

    #return None if predictions are nan because predict method not working well for this model
    if np.isnan(predictions.values).any():
        return predictions, None, None, None, None
    
    #find the error
    holtModelToUseTrain = ExponentialSmoothing(endog=train_data,seasonal_periods=period,trend='add',seasonal='add',\
                                       damped=damped).fit()
    mape,error_start_date,error_end_date=mean_absolute_percentage_error(holtModelToUseTrain, test_data)
    mase=mean_absolute_scaled_error(train_data, test_data, holtModelToUseTrain, period)
        
    return predictions, mape, mase, error_start_date, error_end_date

def mean_absolute_scaled_error(train_data, test_data, model, period):
    """
    Calculates the predictions for the holt winters model, predictions for a naive model, and true values for the 
    test data time length of the data 
    Returns the mean absolute scaled error of the model
    """
    length_train_data=len(train_data[train_data.columns[0]])
    length_test_data=len(test_data[test_data.columns[0]])

    predictions=model.predict(length_train_data+1, length_train_data+length_test_data)
    #make sure predictions doesn't have nans
    if np.isnan(predictions.values).any():
        return np.nan
    pred_naive = get_naive_predictions(train_data, length_test_data, period)
    true_vals=test_data[test_data.columns[0]]

    mase =  mean_absolute_error(true_vals, predictions) / mean_absolute_error(true_vals, pred_naive)

    return mase

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

def determine_lower_mae(model1, model2, test_data):
    """
    Determines which model has a lower mean absolute error and returns True if model 1 and False
    if model 2
    """
    #get the predictions of the length of the test data and true values for the test data 
    predictions_model1=model1.predict(test_data.index[0], test_data.index[-1])
    predictions_model2=model2.predict(test_data.index[0], test_data.index[-1])
    true_vals=test_data[test_data.columns[0]]

    #get the error values
    mae_model1=mean_absolute_error(true_vals, predictions_model1)
    mae_model2=mean_absolute_error(true_vals, predictions_model2)

    return mae_model1<=mae_model2
    
def mean_absolute_percentage_error(model, test_data):
    """
    Calculates the mean absolute percentage error and returns it along with the start and end dates for the 
    test data
    """ 
    predictions=model.predict(test_data.index[0], test_data.index[-1])
    true_vals=test_data[test_data.columns[0]]
    start_date=pd.Timestamp.date(test_data.index[0])
    end_date=pd.Timestamp.date(test_data.index[-1])
    absolute_errors=[]
    
    #just skip if 0 to avoid divide by 0 errors
    count=0 
    for true_val in true_vals:
        if true_val==0:
            continue
        absolute_errors.append(np.abs((true_val-predictions[count])/true_val))
        count+=1
    return np.mean(absolute_errors)*100, start_date, end_date

def get_train_test_data(data, period, freq):
    """
    Returns a new dataframe with about two thirds of the data to use for training the holt winters model, and the other 
    one third for the testing data
    Need to make sure train data has at least 2 periods because that is what the holt winters method requires
    """
    index_two_thirds=int((len(data)/3)*2)
    #if the 2/3rds point is less than 2 periods, make it so it is 2 periods (still should be at least one data point for test_data)
    if index_two_thirds<2*period:
        index_two_thirds=2*period

    #make the train data testing two thirds of the data
    train_data = pd.DataFrame(index=data.index[0:index_two_thirds], columns=["data"])
    train_data["data"] = data["data"][0:int(index_two_thirds)]
    train_data.index = pd.DatetimeIndex(train_data.index.values, freq=freq)
    
    test_data = pd.DataFrame(index=data.index[index_two_thirds:], columns=["data"])
    test_data["data"] = data["data"][int(index_two_thirds):]
    test_data.index = pd.DatetimeIndex(test_data.index.values, freq=freq)
    return train_data, test_data
