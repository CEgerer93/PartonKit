#!/usr/bin/python3

from functools import reduce
import re

# Make a tuple
def as_tuple(s):
    ints=re.compile('-?\d') # Match group of digits that could possibly be preceeded by a minus sign
    return tuple(int(i) for i in ints.findall(s))
                
# Average a list
def avg(d):
    return reduce(lambda a, b: a + b, d) / len(d)

# Global function to make zsep string
def makeAdatZSep(z):
    s=''
    for i in range(0,z):
        s+='3'
    return s

# Global function to cast momentum to string
def mom2Str(x,y,z):
    return str(x)+str(y)+str(z)

def toStr(vec,sp='.'):
    vstr=''
    for i in vec.split(sp):
        vstr+=str(i)
    return vstr
    

# Global function to make mom_type
def mom_type(m):
    tmp=[]
    [tmp.append(i) for i in m]
    for n, m in enumerate(tmp):
        if int(m) < 0:
            m = str(int(m)*-1)

    # Now sort and return
    tmp.sort(reverse=True)
    return "%s%s%s"%(tmp[0],tmp[1],tmp[2])            

# Global function to determine LG
def LG(m):
    m=as_tuple(m)

    # Make all elements non-negative
    
    
    numZero = m.count(0)
    uniqNonZero=[]
    for i in m:
        if i not in uniqNonZero and -1*i not in uniqNonZero and i != 0:
            uniqNonZero.append(i)
            
    if numZero == 3: return 'G_{1g}'
    elif numZero == 2: return 'Dic_4'
    elif numZero == 1 and len(uniqNonZero) == 1: return 'Dic_2'
    elif numZero == 0 and len(uniqNonZero) == 1: return 'Dic_3'
    elif numZero == 1 and len(uniqNonZero) == 2: return 'C4_{nm0}'
    elif numZero == 0 and len(uniqNonZero) == 2: return 'C4_{nnm}'
    else:
        return 'C2_{nmp}'
    
    
