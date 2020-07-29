#import statements
import pandas as pd
import numpy as np
import numpy.ma as ma
import sesd
from datetime import datetime
import statistics
import statsmodels.tsa.stattools
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.statespace
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import math
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings
warnings.simplefilter('ignore')


def forecast(data, period, freq, numPointsPredict, criticalVal=1.645):
    """
    Given data should be indexed dataframe with single column that contains utility data
    Returns the predictions of the sarima model with the best aic value, 
    the mean absolute percent error and mean absolute scaled error, the start and end dates for the test data
    None returned for everything if predictions returns nans which means that sarima couldn't predict for that function
    Default for period is for monthly data, default for crtical value is for alpha=.05
    2 full periods of data required for sarima model and there's a check for this in tool
    """

    col=data.columns[0] #ignore the rest of the columns if more than one
    ts = data[col]

    d = 0
    sp = 0
    sq = 0
    sd = 1 #always one because seasonal differencing always applied
    
    #do seasonal differencing
    data_differenced = ts - ts.shift(period)
    data_differenced = data_differenced[period:]

    #get the acf/pacf values and bounds from the acf and pacf plots
    acfVals,upper_significance_acf,lower_significance_acf = find_acf_or_pacf_and_bounds(data_differenced, len(data),\
                                                                                  criticalVal, pacf=False) 
    pacfVals,upper_significance_pacf,lower_significance_pacf = find_acf_or_pacf_and_bounds(data_differenced, len(data),\
                                                                                  criticalVal, pacf=True)
    
    #if still too many are significant or the data isn't stationary, do first difference
    #not doing second order differencing NOTE: should I try 2nd order difference
    if too_many_significant(data_differenced,len(ts),criticalVal,acfVals,upper_significance_acf,lower_significance_acf)\
            or determine_stationary(data_differenced):
        data_differenced = data_differenced - data_differenced.shift(1)
        data_differenced = data_differenced[1:]
        d=1 #set d to 1 because did first difference
    
    #collect the hyperparameters
    potential_qs=find_q(data_differenced, acfVals,upper_significance_acf,lower_significance_acf, period)
    potential_ps=find_p(data_differenced, pacfVals,upper_significance_pacf,lower_significance_pacf, period)
    sq=find_seasonal_q_or_p(acfVals,upper_significance_acf,lower_significance_acf, period)
    sp=find_seasonal_q_or_p(pacfVals,upper_significance_pacf,lower_significance_pacf, period)

    pattern = len(potential_ps)*len(potential_qs)

    modelSelection = pd.DataFrame(index=range(pattern), columns=["model", "aic"])
    train_data, test_data=get_train_test_data(data, period, freq)

    #run through possible models with varying p and q values 
    num = 0
    for p in range(0, len(potential_ps)):
        for q in range(0, len(potential_qs)):
            sarima = sm.tsa.SARIMAX(
                endog=train_data, order=(potential_ps[p],d,potential_qs[q]), 
                seasonal_order=(sp,sd,sq,period), 
                enforce_stationarity = False, 
                enforce_invertibility = False,
                freq=freq).fit()
            modelSelection.loc[num, "model"] = [[potential_ps[p], d, potential_qs[q]], [sp, sd, sq, period]]
            modelSelection.loc[num,"aic"] = sarima.aic
            num = num + 1

    #take the models with the top 5 lowest aic values to find the lowest mse error from with nested cv
    sorted_df_top5=modelSelection.sort_values(by='aic').head(5)
    errors=[]
    #the errors corresponds with the index in the sorted dataframe so can find the lowest error more easily
    for model in sorted_df_top5["model"].values:
        errors.append(nestedCV(model, data, freq))

    #get the best sarima model
    best_parameters=sorted_df_top5["model"].values[errors.index(min(errors))]
    best_sarima = sm.tsa.SARIMAX(
                endog=data, order=best_parameters[0], 
                seasonal_order=best_parameters[1], 
                enforce_stationarity = False, 
                enforce_invertibility = False,
                freq=freq).fit()

    #get the predictions for the chosen model
    predictions = sarima.predict((len(ts)-1), (len(ts)-1+numPointsPredict))

    #make sure the predictions dont have negatives- instead change to zeros(simple method)
    predictions[predictions<0]=0

    #return None if predictions are nan because predict method not working well for this model
    #methods to find the best sarima mostly don't choose the model with nans for predictions
    if np.isnan(predictions.values).any():
        return predictions, None, None, None, None

    #get the errors
    best_sarima_train = sm.tsa.SARIMAX(
            endog=train_data, order=best_parameters[0], 
            seasonal_order=best_parameters[1], 
            enforce_stationarity = False, 
            enforce_invertibility = False,
            freq=freq).fit()

    mape,error_start_date,error_end_date=mean_absolute_percentage_error(best_sarima_train, test_data)
    mase=mean_absolute_scaled_error(train_data, test_data, best_sarima_train, period)

    return predictions, mape, mase, error_start_date, error_end_date

def mean_absolute_scaled_error(train_data, test_data, model, period):
    """
    Calculates the predictions for the sarima model, predictions for a naive model, and true values for the test data time length
    of the data 
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

def mean_absolute_percentage_error(sarima, test_data):
    """
    Calculate the mean absolute percentage error and return it along with the start and end dates
    for the test data
    """
    predictions=sarima.predict(test_data.index[0], test_data.index[-1])
    #make sure predictions doesn't have nans, just return nan if it does
    if np.isnan(predictions.values).any():
        return np.nan,None,None
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
    return np.mean(absolute_errors)*100,start_date,end_date

def get_train_test_data(data, period, freq):
    """
    Returns a new dataframe with two thirds of the data to use for training the sarima model, and the other 
    one third for the testing data
    If two thirds of the data is less than 2 periods, makes the train data 2 periods
    """
    index_two_thirds=int((len(data)/3)*2)
    #if the 2/3rds point is less than 2 periods, make it so it is 2 periods (still should be at least one data point for test_data)
    if index_two_thirds<2*period:
        index_two_thirds=2*period

    #make the train data testing only two thirds or 2 periods of the data
    train_data = pd.DataFrame(index=data.index[0:index_two_thirds], columns=["data"])
    train_data["data"] = data["data"][0:int(index_two_thirds)]
    train_data.index = pd.DatetimeIndex(train_data.index.values, freq=freq)
    
    test_data = pd.DataFrame(index=data.index[index_two_thirds:], columns=["data"])
    test_data["data"] = data["data"][int(index_two_thirds):]
    test_data.index = pd.DatetimeIndex(test_data.index.values, freq=freq)
    return train_data, test_data
    
def determine_stationary(ts, p_thres=.05):
    """
    Returns true if reject the null hypothesis and data is stationary and false otherwise
    """
    return statsmodels.tsa.stattools.adfuller(ts)[1]<=p_thres

def find_seasonal_q_or_p(pacf_or_acf,upper_significance,lower_significance,period=12):
    """
    Checks if the lag is signficant at period-th lags. Can work for acf or pacf, just needs to pass the 
    right acf/pacf numbers and bounds
    """
    seasonal_val=0
    #for every lag on the period that is significant, add one to the q or p value. 
    #Once a period is not significant, break
    for i in range(period,len(pacf_or_acf), period):
        if pacf_or_acf[i]>upper_significance[i] or pacf_or_acf[i]<lower_significance[i]:
            seasonal_val+=1
        else:
            break
    return seasonal_val

def find_p(data_differenced, pacfVals, upper_significance,lower_significance, period, fract_significant=2/3):
    """
    Returns the potential p values for the data given based on looking at the pacf values from the differenced data
    The potential p values returned need to be less than the period
    """
    indices_significant=[]
    #skip the first acf value (lag 0) since it's normally high and find the indices of all other significant values
    for i in range(1, len(pacfVals)):
        if pacfVals[i]>upper_significance[i] or pacfVals[i]<lower_significance[i]:
            indices_significant.append(i)
    
    potential_ps=[]
    #the p value cannot be greater than the period, get the 5 most likely values for the p value
    while len(potential_ps)<5 and indices_significant!=[]:
        #only consider the index if it is less than the period (don't want to have a very high p)
        if indices_significant[-1]<period:
            potential_ps.append(indices_significant[-1])
        indices_significant.pop(-1)
        
    #if no potential ps, return 0th position
    if potential_ps==[]:
        return [0]
    
    return potential_ps
    
def find_q(data_differenced, acfVals,upper_significance,lower_significance, period,fract_significant=2/3):
    """
    Returns the potential q values for the data given based on looking at the acf values from the differenced data
    """
    indices_significant=[]
    #skip the first acf value since it's normally high and find the indices of all other significant values
    for i in range(1, len(acfVals)):
        if acfVals[i]>upper_significance[i] or acfVals[i]<lower_significance[i]:
            indices_significant.append(i)
    
    #doesn't need to be less than the period, but should be
    potential_qs=[]
    while len(potential_qs)<4 and indices_significant!=[]:
        #only consider the index if it is less than the period (don't want to have a very high q value)
        if indices_significant[-1]<period:
            potential_qs.append(indices_significant[-1])
        indices_significant.pop(-1)
        
    #if no potential qs, return 0th position
    if potential_qs==[]:
        return [0]
        
    return potential_qs
        
def too_many_significant(ts,len_data,criticalVal,acfVals,upper_significance,lower_significance,numSignificantAllowed=5):
    """
    Returns true if the number of significant values is above the given threshold and false if otherwise
    """ 
    num_significant=0
    #determine how many points are significant based on upper and lower bound levels
    for i in range(0, len(acfVals)):
        if acfVals[i]>upper_significance[i] or acfVals[i]<lower_significance[i]:
            num_significant+=1
            
    return num_significant > numSignificantAllowed
 
def find_acf_or_pacf_and_bounds(data, len_data, criticalVal, pacf=False):
    """
    Returns the acf values and the upper and lower significance bounds
    Pacf boolean tells whether to get pacf or acf
    """
    if not pacf:
        vals=statsmodels.tsa.stattools.acf(data, nlags=len(data), fft=False)  #NOTE: fft true or false?
    else:
        vals=statsmodels.tsa.stattools.pacf(data, nlags=len(data)-1)
        
    #find the significance levels to determine how many points outside of it
    upper_significance=[]
    lower_significance=[]

    for i in range(0, len(vals)):
        significance=criticalVal/math.sqrt(len_data-i)
        upper_significance.append(significance)
        lower_significance.append(significance*-1)
        
    return vals, upper_significance, lower_significance

def nestedCV(model_params, train_dataset, freq, k = 3):
    """
    Performs nested cv on the given dataset with the given hyper parameters
    """
    train_dataset.index = pd.DatetimeIndex(train_dataset.index.values, freq=freq)
    tscv = TimeSeriesSplit(n_splits = k)
    mses = []

    #loop through the split data
    for train_index, val_index in tscv.split(train_dataset):
        cv_train, cv_val = train_dataset.iloc[train_index], train_dataset.iloc[val_index]
        sarima = sm.tsa.SARIMAX(
                            endog=cv_train, order= model_params[0], # for SARIMAX, it means (p,d,q), 
                            seasonal_order= model_params[1], # for SARIMAX, it means (sp,sd,sq,12), 
                            enforce_stationarity = False, 
                            enforce_invertibility = False).fit()

        predictions = sarima.predict(cv_val.index.values[0], cv_val.index.values[-1])
        true_values = cv_val.values
        #sometimes returns nan for predictions, but just ignore
        if not np.isnan(predictions.values).any():
            mse = math.sqrt(mean_squared_error(true_values, predictions.values))
            mses.append(mse)

    return np.mean(mses)