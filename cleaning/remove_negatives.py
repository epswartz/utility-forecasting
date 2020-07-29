def make_negatives_0(ts):
    """
        makes all the negatives 0s in the time series so that they don't hurt calculates of the mean and std
        (since negatives are errors anyway)
    """
    count=0
    for ele in ts:
        if ele<0:
            ts[count]=0
        count+=1
