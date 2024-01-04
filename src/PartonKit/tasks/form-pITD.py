#!/usr/bin/python3

import optparse
import h5py
from util.pitd_util import pitd

usage = "Usage: %prog [options] "
parser = optparse.OptionParser(usage);

parser.add_option("-e", "--ensem", type="str", default='',
                  help='Ensemble name (default='')')
parser.add_option("-I", "--insertion", type="int", default="-1",
                  help='Gamma matrix to consider (default = -1)')
parser.add_option("-c", "--cfgs", type="int", default=1,
                  help='Number of configurations (default = 1)')
parser.add_option("-L", "--Lx", type="int", default=32,
                  help='Spatial size')
parser.add_option("-f","--pitdAmplitudesH5",type="str", default='',
                  help='H5 file containing pseudo ITD results (default = '')')
parser.add_option("-m","--matelemsH5",type="str", default='',
                  help='H5 file containing matelem fit results (default = '')')
parser.add_option("-T","--tFitRange",type="str", default='x.x',
                  help='T fit range <tmin>.<tmax> (default = "x.x")')
parser.add_option("-a", action="store_true", default=False, dest="adat",
                  help='Form pitd from adat constructed dat files')
parser.add_option("-P", action="store_true", default=False, dest="PARITY",
                  help='Apply P-symmetry when forming pITD (i.e. conj. neg. ioffe-time data & avg.)')

# Parse the input arguments
(options, args) = parser.parse_args()


# Check if an h5 file has been passed, in case adat constructed dat files are not to be read
if not(options.adat):
    try:
        h=h5py.File(options.pitdAmplitudesH5,'r')
    except:
        raise Exception("Pseudo H5 file not found - adat constructed dat files not to be read")
#-----------------------------------------------------------------------------------------------



pitd=pitd(options.pitdAmplitudesH5,options.matelemsH5,\
          options.cfgs,options.insertion,\
          options.tFitRange.split('.')[0],options.tFitRange.split('.')[1],\
          applyCP=options.PARITY)
pitd.extract()



pitd.reducedPITD('Y',renormWithMatelems=False)
print("Done forming reduced pseudo-ITD for Y")
pitd.reducedPITD('R',nuOrZSqrEqualZeroAmp='Y',renormWithMatelems=False)
print("Done forming reduced pseudo-ITD for R")



##############################
# INITIALIZE HDF5 STORAGE
##############################
parStr='no-parity-applied.'
if options.PARITY: parStr='parity-applied.'

for AMP,DICT in pitd.reducedAmps.items():
    h5File = h5py.File('%s-%s.L-summ_tmin%d-tmax%d.%s_rpITD.%sh5'%\
                       (options.ensem,insertions[options.insertion]['Redstar'],
                        int(options.tFitRange.split('.')[0]),int(options.tFitRange.split('.')[1]),
                        AMP,parStr), 'a')
 
    h5Dirac = h5File.get(insertions[options.insertion]['Redstar'])
    # If the insertion group doesn't exist, make it
    if h5Dirac == None:
        h5Dirac = h5File.create_group(insertions[options.insertion]['Redstar'])
    # Now have "/insertion" in h5 file...
    


    ###############
    # Loop over all p/z keys
    for K,V in DICT.items():
        h5Zsep = h5Dirac.get("%s"%K.split(',')[1])
        # If this zsep group doesn't exist, make it
        if h5Zsep == None:
            h5Zsep = h5Dirac.create_group("%s"%K.split(',')[1])
        
        h5Mom = h5Zsep.get("%s"%K.split(',')[0])
        # If this momentum group doesn't exist, make it
        if h5Mom == None:
            h5Mom = h5Zsep.create_group("%s"%K.split(',')[0])
            # Now have "/insertion/momz" in h5 file...


        #######################
        # Determine Ioffe time
        disp=as_tuple(K.split(',')[1])
        mom=as_tuple(K.split(',')[0].split('_')[0])
        ioffe=ioffeTime(mom,disp,options.Lx)
        #------------------------------------------
    
        for n, component in enumerate([ "Re", "Im" ]):
            # For convenience hold dataset that will be pushed to file based on component
            data=None
            if component == "Re": data=V['real']
            if component == "Im": data=V['imag']
            
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
            h5EnsemMatelem = h5Ensem.create_dataset("pitd", (1,3), 'd')
            h5EnsemMatelem[0,0]=ioffe
            h5EnsemMatelem[0,1]=float(data['avg'])
            h5EnsemMatelem[0,2]=float(data['err'])
            
            # Create the jackknife data structure
            #   -- holding N configs; ioffetime, jackM
            h5JackMatelem = h5Jack.create_dataset("pitd", (options.cfgs,2), 'd')
            for g in range(0,options.cfgs):
                h5JackMatelem[g,0] = ioffe
                h5JackMatelem[g,1] = float(data['jks'][g])
                ############################################################################
