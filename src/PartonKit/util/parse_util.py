#!/usr/bin/python3
####
# Functions to parse different type of pdf fit parameters
####
import numpy as np
import sys

# Pass parameter file, empty parameter array, and pdfType to reader
def reader(f,p,order,pdfType):
    # Read fit parameters from file "f"
    with open(f) as ptr:
        for cnt, line in enumerate(ptr):
            L=line.split(' ')
            # As of 02-23-2021 Fit params include norm, which isnt needed in qval
            if pdfType == 'jam':
                del L[1]

            for n, k in enumerate(order):
                p[k].append(float(L[n]))




# Get the chi2/dof
def chi2(p):
    dumc=0.0
    meanc=np.mean(p['chi2'])
    for g in range(0,ncfg):
        dumc+=np.power(p['chi2'][g]-meanc,2)
    dumc*=((1.0*(ncfg-1))/ncfg)
    



