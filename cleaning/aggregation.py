import pandas as pd
import numpy as np
from datetime import datetime

def aggregate(dataGiven, freq):
    """
    Aggregate the data to the given frequency,
    for cumulative 15 minute data, another function
    may need to be used to aggregate daily
    """

    data=dataGiven.copy()
    col=data.columns[0]

    data = drop_beg_nans_rest_0(data)

    #aggregate the data to the desired frequency
    data.index = pd.to_datetime(data.index)
    aggregated_data = data.groupby(pd.Grouper(freq=freq)).sum()

    return aggregated_data

def drop_beg_nans_rest_0(data):
    """
    Drops nans at the beginning of the given dataframe,
    and sets other nans to zero.
    """

    #find the first entry
    firstEntry=0
    col=data.columns[0]
    for ele in data[col]:
        if not np.isnan(ele):
            break
        firstEntry+=1
    #create data without the beginnings nans
    result=pd.DataFrame(index=data.index[firstEntry:], columns=['data'])
    result['data']=data[col][firstEntry:]
    #make the remaining nans 0s
    result["data"][np.isnan(result["data"])]=0

    return result