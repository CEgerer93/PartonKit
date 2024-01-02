#!/usr/bin/python3

import numpy as np
import h5py


######
# A personalized reader for matrix elements stored in h5 format
######
def reader(h5,cfgs,ztag,ptag,comp,dtype,curr='b_b0xDA__J0_A1pP'):

    ioffeTime = -1
    avgMat = 0.0
    avgMatErr = 0.0
    for g in range(0,cfgs):
        ioffeTime, mat = h5['/%s/%s/%s/jack/%s/%s'%\
                            (curr,ztag,ptag,comp,dtype)][g]
        avgMat += mat
        # # Pack matDict
        # matDict[thisPZKey].append(mat)

                    
    avgMat *= (1.0/cfgs)

    # # Pack avgMatDict
    # avgMatDict.update({thisPZKey: avgMat})

    for g in range(0,cfgs):
        ioffeTime, mat = h5['/%s/%s/%s/jack/%s/%s'%\
                            (curr,ztag,ptag,comp,dtype)][g]
        avgMatErr += np.power( mat - avgMat, 2)
        
    avgMatErr = np.sqrt( ((1.0*(cfgs-1))/cfgs)*avgMatErr )
    

    return ioffeTime, avgMat, avgMatErr
