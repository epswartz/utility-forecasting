import pandas as pd
import numpy as np
import datetime
import calendar
import warnings
warnings.simplefilter('ignore')

def forecast(data, period, freq, numPointsPredict):
    """
    Forecasts the given data with the naive model and returns the predictions as an indexed dataframe
    as well as the mean absolute percent error and the start and end dates for finding that error
    Given at least 2 periods of data
    """
    data=data.copy()
    data.index = pd.DatetimeIndex(data.index.values, freq=freq)
    ts=data[data.columns[0]] #only look at the first column of the data
    train_data, test_data=get_train_test_data(data, period, freq)

    #get a full period of predictions using the train data then find the predictions for the times of the test data
    full_period_predictions_train=get_full_period_predictions(train_data, period)
    predictions_test=get_predictions(full_period_predictions_train, len(test_data["data"]), period)

    #get the mape for the train and test data
    mape,error_start_date,error_end_date=mean_absolute_percentage_error(predictions_test, test_data)

    #get a full period of predictions using all the data then find numPointsPredict predictions 
    full_period_predictions=get_full_period_predictions(data, period)
    predictions=get_predictions(full_period_predictions, numPointsPredict, period)

    return predictions.iloc[:, 0],mape,error_start_date,error_end_date

def mean_absolute_percentage_error(predictions_df, test_data):
    """
    Calculates the mean absolute percentage error given the predictions and the test data
    and returns it along with the start and end dates for the test data
    """ 
    true_vals=test_data[test_data.columns[0]]
    predictions=predictions_df["data"][1:] #ignore the first prediction because repeat
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

def get_predictions(full_period_predictions, numPointsPredict, period):
    """
    Returns numPointsPredict of predictions from looping over the full period of predictions
    which serves as a model for this forecasting method
    """
    lastDate=full_period_predictions['date'][0]
    lastDateIndex=0
    #find the most recent date to start at
    for i in range(0,len(full_period_predictions['data'])):
        if full_period_predictions["date"][i]>lastDate:
            lastDate=full_period_predictions["date"][i]
            lastDateIndex=i

    #create index for predictions starting at lastDate with numPointsPredict length
    dates=get_indices(lastDate,numPointsPredict)
    predictions=pd.DataFrame(index=dates, columns=['data'])

    #start predicting from the last point onwards
    for i in range(0,numPointsPredict+1): #+1 because include predict for lastDate + numPointsPredict
        predictions['data'][i]=full_period_predictions["data"][lastDateIndex]
        #if you reach the end of the period, reset to beginning of full_period_predictions
        lastDateIndex+=1
        if lastDateIndex>=period:
            lastDateIndex=0
    return predictions

def get_indices(startDate,numDates):
    """
    Returns a list of dates starting with start date that is numDates+1 long (since including startDate)
    Each date is the last date of that month
    This only works with monthly, but we are only forecasting with monthly
    """
    dates=[]
    curDate=startDate
    for i in range(0,numDates+1):#+1 because repeats start date
        #get the last day of the month for that year
        last_day_of_month = calendar.monthrange(curDate.year,curDate.month)[1]
        dates.append(datetime.datetime(curDate.year,curDate.month, last_day_of_month))
        newYear=curDate.year
        newMonth=curDate.month+1
        if newMonth>12:
            newMonth=1
            newYear+=1
        curDate=datetime.datetime(newYear,newMonth,1)
    return dates


def get_full_period_predictions(data, period):
    """
    Gets the predictions for the simple model by averaging each data point (can't assume given full periods)
    Returns a dataframe the same size as a full period- each period point is an average of the other period
    points at the same spot so model will stay the same forecasting outward
    full period predictions has date column with the last date used to average that month/day/week
    Data will have at least 1 period
    Basically serves as the model for this modeling method
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
    return full_period_predictions


def get_train_test_data(data, period, freq):
    """
    Returns a new dataframe with about two thirds of the data to use for averaging, and the other 
    one third for the testing data
    """
    index_two_thirds=int((len(data)/3)*2)
    dataCol=data.columns[0]
    #make the train data testing two thirds of the data
    train_data = pd.DataFrame(index=data.index[0:index_two_thirds], columns=["data"])
    train_data["data"] = data[dataCol][0:int(index_two_thirds)]
    train_data.index = pd.DatetimeIndex(train_data.index.values, freq=freq)
    
    test_data = pd.DataFrame(index=data.index[index_two_thirds:], columns=["data"])
    test_data["data"] = data[dataCol][int(index_two_thirds):]
    test_data.index = pd.DatetimeIndex(test_data.index.values, freq=freq)
    return train_data, test_data

