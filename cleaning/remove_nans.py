import pandas as pd
import numpy as np

def drop_beg_nans_rest_0(data):
    """
    Drops nans at the beginning of the given dataframe, and sets other nans to zero.
    Doesn't modify the given data
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
