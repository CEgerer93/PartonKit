#!/usr/bin/python3

from scipy.spatial.transform import Rotation as R
import numpy as np
import sys,optparse

# Parse command line options
usage = "usage: %prog [options] "
parser = optparse.OptionParser(usage) ;

parser.add_option("-g", "--littleGroup", type="str", default='',
                  help='Little group (default = '')')
parser.add_option("-m", "--momType", type="str", default='',
                  help='Momentum type as n0n,-n00,etc. (default='')')

(options, args) = parser.parse_args()


# Make dictionary with Rref Euler angles based on little group
Rref={'Dic4': R.from_euler('zyz', [0,0,0]),\
      'Dic2': R.from_euler('zyz', [np.pi/2,np.pi/4,-np.pi/2]),\
      'Dic3': R.from_euler('zyz', [np.pi/4,np.arccos(1/np.sqrt(3)),0]),\
      'C4mn0': R.from_euler('zyz', [np.pi/2,np.arccos(2/np.sqrt(5)),0]),\
      'C4mnn': R.from_euler('zyz', [-3*np.pi/4,-np.arccos(np.sqrt(2/3)),0])}


# # Print rotation matrices Rref for each little group's Euler angles
# for k,v in Rref.items():
#     print(k)
#     print(v.as_matrix())





# Make dictionary with Rlat Euler angles based on little group and specific momentum
Rlat={'Dic4': {'n00': R.from_euler('ZYZ', [0,np.pi/2,0]),\
               '0n0': R.from_euler('ZYZ', [np.pi/2,np.pi/2,0]),\
               '00n': R.from_euler('ZYZ', [0,0,0]),\
               '-n00': R.from_euler('ZYZ', [np.pi,np.pi/2,0]),\
               '0-n0': R.from_euler('ZYZ', [-np.pi/2,np.pi/2,0]),\
               '00-n': R.from_euler('ZYZ', [0,np.pi,0])},\
      'Dic2': {'nn0': R.from_euler('ZYZ', [0,np.pi/2,0]),\
               '0nn': R.from_euler('ZYZ', [0,0,0]),\
               'n0n': R.from_euler('ZYZ', [2*np.pi,0,3*np.pi/2]),\
               'n-n0': R.from_euler('ZYZ', [7*np.pi/2,np.pi/2,0]),\
               '0n-n': R.from_euler('ZYZ', [0,-np.pi,0]),\
               '-n0n': R.from_euler('ZYZ', [2*np.pi,0,np.pi/2]),\
               '-nn0': R.from_euler('ZYZ', [0,-np.pi/2,0]),\
               '0-nn': R.from_euler('ZYZ', [2*np.pi,0,np.pi]),\
               'n0-n': R.from_euler('ZYZ', [2*np.pi,np.pi/2,3*np.pi/2]),\
               '-n-n0': R.from_euler('ZYZ', [7*np.pi/2,np.pi/2,np.pi]),\
               '0-n-n': R.from_euler('ZYZ', [2*np.pi,-np.pi,np.pi]),\
               '-n0-n': R.from_euler('ZYZ', [2*np.pi,-np.pi/2,np.pi/2])}}


# for k,v in Rlat[options.littleGroup].items():
#     print(k)
#     print(v.as_matrix())




# Multiply Rlat & Rref as arrays
res=Rlat[options.littleGroup][options.momType].as_matrix()@Rref[options.littleGroup].as_matrix()
resFromMat=R.from_matrix(res)
resAsMat=resFromMat.as_matrix()


# Print the resulting Euler angles
print(resFromMat.as_euler('ZYZ'))



alpha=np.arctan2(resAsMat[1,2],resAsMat[0,2])
beta=np.arccos(resAsMat[2,2])
gamma=np.arctan2(-resAsMat[2,1],resAsMat[2,0])

print("Alpha = %.8f"%(alpha/np.pi))
print("Beta  = %.8f"%(beta/np.pi))
print("Gamma = %.8f"%(gamma/np.pi))
