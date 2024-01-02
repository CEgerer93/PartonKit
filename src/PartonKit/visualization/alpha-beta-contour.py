#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.special as spec
import scipy.integrate as integrate
# import scipy.interpolate.griddata
# import mathplotlib.tri as tri
import pylab
import sys,optparse
from collections import OrderedDict

from mpl_toolkits import mplot3d

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-p", "--paramsFile", type="str", default="",
                  help='Jacobi fit params associated w/ alpha,beta scan (default = "")')
parser.add_option("-a", "--jacobiCorrections", type="str", default="",
                  help='Jacobi approx of pITD corrections; <numLT>.<numAZ>.<numT4>.<numT6>.<numT8>.<numT10> (default = "")')
parser.add_option("-t", "--pdfType", type="str", default="",
                  help='Params associated w/ PDF of type valence (v) or plus (+) (default = "")')
parser.add_option("-A", "--bestAlpha", type="float", default=0.0,
                  help='Best fit alpha (default=0.0)')
parser.add_option("-B", "--bestBeta", type="float", default=0.0,
                  help='Best fit beta (default=0.0)')


# Parse the input arguments
(options, args) = parser.parse_args()
params=options.paramsFile

# Instantiate figures to show correlations of pdf fit parameters
fig_ab=plt.figure(figsize=(12,10))
ax_ab=fig_ab.gca()


numLT = int(options.jacobiCorrections.split('.')[0])
numAZ = int(options.jacobiCorrections.split('.')[1])
numT4 = int(options.jacobiCorrections.split('.')[2])
numT6 = int(options.jacobiCorrections.split('.')[3])
numT8 = int(options.jacobiCorrections.split('.')[4])
numT10 = int(options.jacobiCorrections.split('.')[5])

corrStart=1
if options.pdfType == '+':
    corrStart=0

fitParams={'L2': [], 'chi2': [], 'L2/dof': [], 'chi2/dof': [], '\\alpha': [], '\\beta': []}
paramOrder=['L2', 'chi2', 'L2/dof', 'chi2/dof', '\\alpha', '\\beta']
for l in range(0,numLT):
    fitParams.update({'C^{lt}_%d'%l: []})
    paramOrder.append('C^{lt}_%d'%l)
for a in range(corrStart,numAZ+corrStart):
    fitParams.update({'C^{az}_%d'%a: []})
    paramOrder.append('C^{az}_%d'%a)
for t in range(corrStart,numT4+corrStart):
    fitParams.update({'C^{t4}_%d'%t: []})
    paramOrder.append('C^{t4}_%d'%t)
for s in range(corrStart,numT6+corrStart):
    fitParams.update({'C^{t6}_%d'%s: []})
    paramOrder.append('C^{t6}_%d'%s)
for u in range(corrStart,numT8+corrStart):
    fitParams.update({'C^{t8}_%d'%u: []})
    paramOrder.append('C^{t8}_%d'%u)
for v in range(corrStart,numT10+corrStart):
    fitParams.update({'C^{t10}_%d'%v: []})
    paramOrder.append('C^{t10}_%d'%v)



with open(params) as ptr:
    for cnt, line in enumerate(ptr):
        
        # Capture the line and remove spaces
        L=line.split(' ')
        for n, k in enumerate(paramOrder):
            # if n%5 == 0:
            fitParams[k].append(float(L[n]))
        


a=np.array(fitParams['\\alpha'])
b=np.array(fitParams['\\beta'])
chi=np.array(fitParams['L2/dof'])

colormap = plt.get_cmap("YlOrRd")

slicer=30

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter3D(a[::slicer],b[::slicer],chi[::slicer])
ax.set_zlim([0,30])
ax.azim=-120
ax.elev=30
# ax.set_zlim([-230,-50])

ax.plot3D(options.bestAlpha,options.bestBeta,min(chi[::slicer]),'rx')
plt.show()
sys.exit()



ai=np.linspace(min(a),max(a),500)
bi=np.linspace(min(b),max(b),500)


A, B = np.meshgrid(ai,bi)


C=np.zeros((500,500))
for a in range(0,len(ai)):
    for b in range(0,len(bi)):
        # key=(ai[a],bi[b])
        # C[a,b] = chi[b+len(bi)*a]
        C[b,a] = chi[b+len(bi)*a]
        

print(C[0,0])


# print("CHECK = %.7f"%chi2Dict.get((-0.184,2.684)))


# Set up the figures
plt.rc('text', usetex=True)
plt.rcParams["mathtext.fontset"]="stix"

# cs=ax_ab.contour(A,B,C,levels=np.linspace(15,2000,300))
cs=ax_ab.contour(A,B,C,levels=np.linspace(1,50,25000))


# ax_ab.scatter(cDict['alpha'],cDict['beta'],c=cDict['chi2'],cmap='hot',s=1)


# Scatter for alpha vs. beta
ax_ab.set_xlabel(r'$\alpha$',fontsize=16)
ax_ab.set_ylabel(r'$\beta$',fontsize=16)
# ax_ab.set_xlim([-0.25,-0.15])
plt.colorbar(cs)

ax_ab.plot(options.bestAlpha,options.bestBeta,'ro')

# # # Save all the figures
# # fig_ab.savefig(options.fit4Params.split('.')[0]+"."+options.fit4Params.split('.')[1]+".alpha-beta.png",dpi=400)

out=options.paramsFile.replace('SCAN.txt','alpha-beta.scan')
fig_ab.savefig(out+".pdf",dpi=400)

plt.show()
