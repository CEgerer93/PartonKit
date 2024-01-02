#!/usr/bin/python3

import sys,optparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
import pylab # to save figures to file
from collections import OrderedDict

from util.pitd_util import pgitd
from util.fit_util import SR

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-f","--pgitdAmplitudesH5",type="str", default='',
                  help='H5 file containing pseudo-GITD results - numerator of rpgitd (default = '')')
parser.add_option("-F","--denomPseudoH5",type="str", default='',
                  help='H5 file containing pseudo ITD results - denominator of rpitd\n If empty, numerPseudoH5 is used for denom (default = '')')
parser.add_option("-w", action="store_true", default=False, dest="denomPseudoH5WithoutSVD",
                  help='Denom pseudo-ITD H5 obtained without SVD [e.g. old way unpol. pITD was made]')
parser.add_option("-n", "--normMom", type="str", default='',
                  help='Momentum of rpGITD denominator X.X.X (default = '')')
parser.add_option("-z", "--forwardDispRange", type="str", default='0.8',
                  help='Displacements to fetch from forward pITD database needed to form pGITD. Note that PDFs were analyzed by averaging over +/-z, so if desired pGITD should be from z\in[-10,10], then this option should read 0.10\n Ignored if "-w" is omitted (default = 0.8)')
parser.add_option("-a", "--selectAmp", type="str", default='',
                  help='Form rpGITD for a select amplitude; all if not passed (default = '')')
parser.add_option("--selectPfPi", type="str", default='',
                  help='Form rpGITD for a select pf_pi combo [f.f.f/i.i.i]; all if not passed (default = '')')
parser.add_option("-e", "--ensem", type="str", default='',
                  help='Ensemble name (default='')')
parser.add_option("-I", "--insertion", type="int", default="-1",
                  help='Forward Gamma matrix to consider (default = -1)')
parser.add_option("--offFwdInsertion", type="int", default="-1",
                  help='Int denoting combination of \gamma_x,\gamma_y,\gamma_4 for GPDs (default = -1)')
parser.add_option("-c", "--cfgs", type="int", default=1,
                  help='Number of configurations (default = 1)')
parser.add_option("-L", "--Lx", type="int", default=32,
                  help='Spatial size')
parser.add_option("-T","--tFitRange",type="str", default='x.x',
                  help='T fit range <tmin>.<tmax> (default = "x.x")')
parser.add_option("-P", action="store_true", default=False, dest="PARITY",
                  help='Apply P-symmetry when forming pGITD (i.e. conj. neg. ioffe-time data & avg.)')
parser.add_option("-D", action="store_true", default=False, dest="doubleRatio",
                  help='Use double ratio in forming pGITD')

# Parse the input arguments
(options, args) = parser.parse_args()


# # h5=None
# # Check if an h5 file has been passed, in case adat constructed dat files are not to be read
# if not(options.adat):
#     try:
#         h5Numer=h5py.File(options.pgitdAmplitudesH5,'r')
#         h5Denom=h5py.File(options.denomPseudoH5,'r')
#     except:
#         raise Exception("Pseudo H5 file not found - adat constructed dat files not to be read")
#         sys.exit()
# #-----------------------------------------------------------------------------------------------



selectPf='' if options.selectPfPi == '' else\
    tuple(p for p in options.selectPfPi.split('/')[0].split('.')) 
selectPi='' if options.selectPfPi == '' else\
    tuple(p for p in options.selectPfPi.split('/')[1].split('.'))

current = insertions[options.insertion]['Redstar']

# Define the momentum used for normalizing a given pGITD to produce a reduced pGITD
nx, ny, nz = tuple(int(n) for n in options.normMom.split('.'))


# Suffix for if parity is applied or not
parStr='no-parity-applied.'
if options.PARITY: parStr='parity-applied.'


##########################################################
# Extract all pgitd data for all pf/pi/zsep combinations #
##########################################################
PGITDs=pgitd(options.pgitdAmplitudesH5,options.cfgs,options.offFwdInsertion,\
             options.tFitRange.split('.')[0],options.tFitRange.split('.')[1])
PGITDs.extract(extAmp=None if options.selectAmp=='' else options.selectAmp,\
               extMom=None if selectPi=='' and selectPf=='' else\
               'pf%s_pi%s'%(''.join(selectPf),''.join(selectPi)))
                                           
# Optionally apply parity in z
if options.PARITY: PGITDs.applyParity()
#--------------------------------------------------------------------------------------------


##########################################################
# Determine how normalizing pseudo-ITD should be handled #
##########################################################
if options.denomPseudoH5WithoutSVD:
    pitd={}
    
if not options.denomPseudoH5WithoutSVD:
    pitd=pitd(options.denomPseudoH5,None,\
              options.cfgs,options.insertion,\
              options.tFitRange.split('.')[0],options.tFitRange.split('.')[1])
    pitd.extract()
    # Optionally apply parity in z
    if options.PARITY: pitd.applyParity()
else:
    # Gather all forward displacements to look up
    allForwardDisps = ["00%i"%d for d in range(int(options.forwardDispRange.split('.')[0]),\
                                               int(options.forwardDispRange.split('.')[1])+1)]

    # Need to use fit_util.SR class to access PDF results
    for d in allForwardDisps:
        pitd.update({d: correlator(None,None,None,None,None)})

        
    for pitdDisp, corr in pitd.items():

        # Run over real/imag
        for k,v in {1: 'real', 2: 'imag'}.items():
            # Read all data from forward matelem pitd h5
            corr.amplitude[k] = fit_util.SR(options.denomPseudoH5,options.cfgs,\
                                            "pf%s"%mom2Str(nx,ny,nz),"pi%s"%mom2Str(nx,ny,nz),0,0,\
                                            "zsep%s"%pitdDisp,options.insertion,v,\
                                            tmin=int(options.tFitRange.split('.')[0]),
                                            tmax=int(options.tFitRange.split('.')[1]),\
                                            pdfH5=True)
            # NB pdfH5 =True will neglect passed rowf,rowi (two 0's above) & read pitd in old format
            
            
            # If the momentum or displacement is zero for c[0]
            # then kill the imaginary component, as ioffe time is zero
            if k == 2 and pitdDisp == "000":
                corr.amplitude[k].params['b'] = np.zeros(options.cfgs)

            
            #     if c[0] is None:
            #         continue
            #     else:
            #         # Run over all displacements needed by pGITDs
            #         for forwardDisp in allForwardDisps:
            #             # Run over real/imag
            #             for k,v in {1: 'real', 2: 'imag'}.items():
            #                 # Read all data from pgitdAmplitudesH5 & denomPseudoH5
            #                 c[0].amplitude[k] = fit_util.SR(options.denomPseudoH5,options.cfgs,\
                #                                                 c[1],c[1],0,0,\
                #                                                 forwardDisp,options.insertion,v,\
                #                                                 tmin=int(options.tFitRange.split('.')[0]),
            #                                                 tmax=int(options.tFitRange.split('.')[1]),\
                #                                                 pdfH5=True)
            #                 # NB pdfH5 =True will neglect passed rowf,rowi (two 0's above) & read pitd in old format
            
            
            #                 # If the momentum or displacement is zero for c[0]
            #                 # then kill the imaginary component, as ioffe time is zero
            #                 if k == 2 and ( c[1] == "000" or forwardDisp == "000" ):
            #                     c[0].amplitude[k].params['b'] = np.zeros(options.cfgs)
            # #--------------------------------------------------------------------------------------------

    print(pitd)
    for k in pitd.keys():
        print(pitd[k].amplitude[1].params['b'][:])




if not options.denomPseudoH5WithoutSVD:
    print("%.5f +/- %.5f"%(pitd.amplitudes['Y']['pf000_pi000,zsep000']['real']['avg'],\
                           pitd.amplitudes['Y']['pf000_pi000,zsep000']['real']['err']))
    print("%.5f +/- %.5f"%(pitd.amplitudes['Y']['pf000_pi000,zsep001']['real']['avg'],\
                           pitd.amplitudes['Y']['pf000_pi000,zsep001']['real']['err']))
    # print("%.5f +/- %.5f"%(pitd.amplitudes['Y']['pf000_pi000,zsep00-1']['real']['avg'],\
        #                        pitd.amplitudes['Y']['pf000_pi000,zsep00-1']['real']['err']))
    


# Form the rpITD by combining fit results per jackknife sample of each amplitude
# NO JACKKNIFING HERE!!!
# rpGITD=pitd(boost,boostNoDisp,norm,normNoDisp,boost_negIT,boostNoDisp_negIT,\
#             norm_negIT,normNoDisp_negIT,matStr='b')


############################################################################################################
# FORM THE rpGITD BY COMBINING FIT RESULTS PER JACKKNIFE SAMPLE OF EACH AMPLITUDE - NO JACKKNIFING HERE!!! #
############################################################################################################
PGITDs.reducedPGITD(pitd,doubleRatio=options.doubleRatio,thisAmpInDblRatio='A1')
#-----------------------------------------------------------------------------------------------------------


#################################
# PRINT VALUES OF FORMED rpGITD #
#################################
for a in ['A%i'%i for i in range(1,11)]:
    for c in ['real', 'imag']:
        for key in PGITDs.reducedAmps[a].keys():
            print("%s %s [%s] = %.7f +/- %.7f"%(c,a,key,PGITDs.reducedAmps[a][key][c]['avg'],PGITDs.reducedAmps[a][key][c]['err']))
print("\n")
#-----------------------------------------------------------------------------------------------------------------------------------



##############################
# INITIALIZE HDF5 STORAGE
##############################
dblRatioStr='' if not options.doubleRatio else '-dblRatio'
h5File = h5py.File('%s-%s.L-summ_tmin%d-tmax%d.rpGITD%s.%sh5'%\
                   (options.ensem,current,
                    int(options.tFitRange.split('.')[0]),int(options.tFitRange.split('.')[1]),
                    dblRatioStr,parStr), 'a')

 
h5Dirac = h5File.get(current)
# If the insertion group doesn't exist, make it
if h5Dirac == None:
    h5Dirac = h5File.create_group(current)
# Now have "/insertion" in h5 file...


# Run over the amplitudes
for a,AMP in PGITDs.reducedAmps.items():
    h5Amp = h5Dirac.get(a)
    if h5Amp == None:
        h5Amp = h5Dirac.create_group("%s"%a)
        # Now have amplitude group

    for keyPfPiZ,V in AMP.items():

        pfpi, disp = keyPfPiZ.split(',')
        pf, pi = pfpi.split('_')

        # Compute Ioffe times
        initIoffe = ioffeTime(as_tuple(pi),as_tuple(disp),options.Lx)
        finIoffe = ioffeTime(as_tuple(pf),as_tuple(disp),options.Lx)
        avgIoffeTime=(initIoffe+finIoffe)/2.0

        h5Zsep = h5Amp.get("%s"%disp)
        # If this zsep group doesn't exist, make it
        if h5Zsep == None:
            h5Zsep = h5Amp.create_group("%s"%disp)

        h5Mom = h5Zsep.get("%s"%pfpi)
        # If this momentum group doesn't exist, make it
        if h5Mom == None:
            h5Mom = h5Zsep.create_group("%s"%pfpi)
            # Now have "/insertion/momz" in h5 file...
        
        for n, component in enumerate([ 'real', 'imag' ]):
            # Check to see if ensemble group exists, and make if none
            h5Ensem = h5Mom.get("ensemble/%s"%component)
            if h5Ensem == None:
                h5Ensem = h5Mom.create_group("ensemble/%s"%component)
            
            # Check to see if jack group exists, and make if none
            h5Jack = h5Mom.get("jack/%s"%component)
            if h5Jack == None:
                h5Jack = h5Mom.create_group("jack/%s"%component)
                
            # Now have "/insertion/ensemble/comp/zsep" in h5 file
            # Now have "/insertion/jack/comp/zsep" in h5 file
            
            
            # Create the ensem data structure
            #   -- holding 1 config; ioffeTime, avgM, errM
            h5EnsemMatelem = h5Ensem.create_dataset("pgitd", (1,3), 'd')
            h5EnsemMatelem[0,0]=avgIoffeTime
            h5EnsemMatelem[0,1]=V[component]['avg']
            h5EnsemMatelem[0,2]=V[component]['err']
            
            # Create the jackknife data structure
            #   -- holding N configs; ioffetime, jackM
            h5JackMatelem = h5Jack.create_dataset("pgitd", (options.cfgs,2), 'd')
            for g in range(0,options.cfgs):
                h5JackMatelem[g,0] = avgIoffeTime
                
                if options.denomPseudoH5WithoutSVD:
                    # h5JackMatelem[g,1] = V[component]['jks'].dat[g]
                    h5JackMatelem[g,1] = V[component]['jks'][g]
                else:
                    h5JackMatelem[g,1] = V[component]['jks'][g]
                ############################################################################

h5File.close()
