#import statements
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import numpy.ma as ma
import sesd
from datetime import datetime
import statistics
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
register_matplotlib_converters()

def removeAnomaliesAndImpute(data,doSeasonalAnomaly,doSTDAnomaly,doIQR,period,multIQR=2.5,std_mult=2.5,seasonal_mult=2.5,k=10,sizeWindows=15):
    """
        Driver that runs all of the functions above- takes in a time series and returns the ts with anomalies removed and imputed
        Params:
            data- data index with dates with only one column that has the data,
            doseasonalanomaly&dostdanomaly&doiqr- determine which anomaly detection methods you would like to use
            period- 365 if daily, 12 if monthly, etc,
            multiqr- multiplier for the iqr method, k for knn data imputation, size of windows for std method
            seasonal_mult is multiplier for the seasonal method, std_mult is for the std method multiplier
        Returns: indexed dataframe of the imputed data, (for graphing)returns boolean arrays where seasonal, std, and iqr anomalies were,
        an array of upper bounds for iqr method (returns None for any of these if didn't do that type of anomaly detection) and 
        what percent of the data was marked as anomalies
    """
    col=data.columns[0] #only look at the first columns if more are given
    ts=data[col].to_numpy()

    #initialize variables (make boolean arrays false so if don't do that type of detection, there's no anomalies)
    ts=ts.copy().astype(float)
    seasonal_anomalies, std_anomalies, iqr_anomalies=[False]*len(ts), [False]*len(ts), [False]*len(ts)
    iqr_upper_bounds=None

    if doSeasonalAnomaly:
        #if couldn't do seasonal anomalies returns None (i.e. data not 2 periods), else returns an array with the locations
        #of seasonal anomalies
        seasonal_anomalies=do_seasonal_anomaly_detection(data, ts, period, seasonal_mult)
    if doSTDAnomaly:
        windows = compute_windows(ts, sizeWindows)
        std_anomalies = [is_anom_std(ts, windows[i], ts[i], multiplier=std_mult) for i in range(0, len(ts))]
    if doIQR:
        #conduct the simple iqr method (not rolling) for extreme groups of outliers
        iqr_anomalies, iqr_upper_bounds=simpleIQR(ts, multIQR)

    #create tsAnomaliesNA which has all of the anomalies marked as nan if they occurred in any of iqr, seasonal or std
    #check the boolean arrays which were initilized as falses, so doesn't matter if user chose not to do one method
    tsAnomaliesNA=ts.copy().astype(float)
    for i in range(0, len(ts)):
        #set to nan if anomaly occurred at least one of these
        if seasonal_anomalies[i] or std_anomalies[i] or iqr_anomalies[i]:
            tsAnomaliesNA[i]=np.nan

    #impute the data
    imputed_data=[imputeKNN(tsAnomaliesNA[i], tsAnomaliesNA, i, k) for i in range(0, len(ts))]

    #create dataframe with indices to return with imputed data
    imputed_data_df=pd.DataFrame(index=data.index, columns=['data'])
    imputed_data_df['data']=imputed_data

    #get the percent of anomalies
    percent_anomalies=(count_anomalies(seasonal_anomalies, std_anomalies, iqr_anomalies)/len(seasonal_anomalies))*100

    #if didn't do certain type of anomaly detection, can make that boolean array None so that the tool knows that anomaly 
    #detection didn't happen
    if not doSeasonalAnomaly:
        seasonal_anomalies=None
    if not doSTDAnomaly:
        std_anomalies=None
    if not doIQR:
        iqr_anomalies=None
    return imputed_data_df, seasonal_anomalies, std_anomalies, iqr_anomalies, iqr_upper_bounds, percent_anomalies

def rolling_window(a, window_size):
    """
        Returns a numpy array of arrays, which are the windows
        a is a numpy array
        This method of generating windows also cuts off the first and last `window_size` of data
    """
    shape = a.shape[:-1] + (a.shape[-1] - window_size + 1, window_size)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def compute_windows(a, window_size):
    """
        Returns windows based on given windows size
    """
    windows = rolling_window(a, window_size)

    # We need copies of the windows at the beginning and end, filled out to the complete length of the series.
    windows = windows.tolist() # Easier to do what I want with a python list
    for i in range(0, window_size//2):
        windows = [windows[0]] + windows

    for i in range(0, window_size//2):
        windows = windows + [windows[-1]]

    return windows

def do_seasonal_anomaly_detection(data, ts, period, std_mul):
    """
        Conducts the seasonal anomaly detection on the data and puts nulls in the time series where the anomalies are
        Returns a boolean array of where the anomalies were
    """
    decompose_result = seasonal_decompose(x=data, model='additive', period=period)

    # Extract the residual
    residual = decompose_result.resid
    residual_data = residual.to_numpy()

    #returns the locations of the seasonal anomalies
    return find_anomalies_seasonality(residual_data, ts, std_mul)

def find_anomalies_seasonality(residual, ts, std_mul):
    """
        Standard deviation-based anomaly detection based on seasonality
        Returns a boolean array with the location of the anomalies
    """
    # Calculate mean and standard deviation ignoring NaNs (shouldn't be nans anyway)
    residual_mean = np.nanmean(residual)
    residual_std = np.nanstd(residual)

    anomalies = []

    # Locate anomalies using standard deviation-based method
    for i in range(0, len(residual)):
        if not np.isnan(residual[i]):
            #appends true or false depending on if current element is out of the range
            anomalies.append(abs(residual[i] - residual_mean) > (std_mul * residual_std))
        #can still be nans in the residual after the starting of data
        else:
            anomalies.append(False)

    # Return boolean array indicating location of anomalies
    return anomalies

def is_anom_std(ts, a, element, multiplier, multStd=4):
    """
        Given a window, returns whether or not the element is anomalous, according to std dev of a
        ts- time series with data, a- window, element- current element
        multipler for the normal std method and multStd is for the special method that checks element against neighbors
    """
    #calculate the mean, std and bounds
    a = np.asarray(a)
    m = np.nanmean(a)
    std = np.nanstd(a)
    upper_bound = m + (multiplier * std)
    lower_bound = m - (multiplier * std)

    #find the mean and std of the array after taking away the current value and its 2 direct neighbors
    #this helps find really sharp peaks in time series
    aWithoutEle=np.delete(a,int(len(a)/2))
    aWithoutEleAndNeighbors=np.delete(aWithoutEle,int(len(aWithoutEle)/2))
    aWithoutEleAndNeighbors=np.delete(aWithoutEleAndNeighbors,int(len(aWithoutEleAndNeighbors)/2))
    stdWithoutEleAndNbors=np.std(aWithoutEleAndNeighbors)
    meanWithoutEleAndNbors=np.mean(aWithoutEleAndNeighbors)

    #find outside bounds in normal std method
    if (element > upper_bound) or (element < lower_bound) or (element < 0):
        return True
    #detect large spikes in the data by checking the difference between the element and its direct neighbor and seeing if
    #its outside a range
    elif abs(element-a[int(len(a)/2)-1])>multStd*stdWithoutEleAndNbors or\
            abs(element-a[int(len(a)/2)+1])>multStd*stdWithoutEleAndNbors:
        return True
    #see if element significantly bigger than the mean and std calculated without itself and its 2 direct neighbors
    #attempts to find large spikes with 3 or so elements
    elif element>meanWithoutEleAndNbors+(multStd*stdWithoutEleAndNbors) or\
            element<meanWithoutEleAndNbors-(multStd*stdWithoutEleAndNbors):
        return True
    else:
        return False

def imputeKNN(element, ts, index, k):
    """
        If the element is nan, imputes the data, else just returns the element
    """
    if np.isnan(element):
        return impute(ts, index, k)
    else:
        return element

def impute(ts, index, k):
    """
        Returns a new value for the nan element based on the average of the k nearest neighbors
        ts- has nans where anomalies were so anomalies not used when computing, index- index of current element
    """
    curNumNeighbors=0
    sumNeighbors=0

    #collect the sum of the k nearest neighbors that aren't nan
    #return the average of the k nearest neighbors when k neighbors found
    for i in range(1, len(ts)):
        if index+i<len(ts) and not np.isnan(ts[index+i]):
            curNumNeighbors+=1
            sumNeighbors+=ts[index+i]
            #when you have found k neighbors return the average
            if curNumNeighbors==k:
                return sumNeighbors/k

        if index-i>=0 and not np.isnan(ts[index-i]):
            curNumNeighbors+=1
            sumNeighbors+=ts[index-i]
            #when you have found k neighbors return the average
            if curNumNeighbors==k:
                return sumNeighbors/k

    #if somehow not enough neighbors found (should never happen), just return what you have even if not k neighbors being averaged
    if curNumNeighbors!=0:
        return sumNeighbors/curNumNeighbors
    else:
        return sumNeighbors

def simpleIQR(ts, mult=2.5):
    """
        Simple IQR method to detect very large wide anomalies
        Returns a boolean array where the positions of the anomalies are marked as true and
        Upper bounds of the iqr method for graphing
    """
    q1=np.nanpercentile(ts, 25)
    q3=np.nanpercentile(ts, 75)
    iqr=q3-q1
    #find the locations of the iqr anomalies
    iqr_anomalies=[]
    for i in range(0, len(ts)):
        iqr_anomalies.append(ts[i]>(q3+(mult*iqr)) or ts[i]<(q1-(mult*iqr)))
    upper_bounds = [q3+(mult*iqr)]*len(ts)
    return iqr_anomalies, upper_bounds

def count_anomalies(bool_arr_1, bool_arr_2, bool_arr_3):
    """
    Given 3 boolean arrays of all the same size with True representing where the anomalies are in the ts, 
    returns the number of anomalies that occur at any of the arrays 
    """
    result=[False]*len(bool_arr_1)
    for i in range(0, len(bool_arr_1)):
        if bool_arr_1[i] or bool_arr_2[i] or bool_arr_3[i]:
            result[i]=True
    return sum(result)