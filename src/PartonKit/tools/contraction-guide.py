#!/usr/bin/python3

import optparse
from util.pgitd_util import pgitd

# Parse command line options
usage = "usage: %prog [options] "
parser = optparse.OptionParser(usage) ;

parser.add_option("-x", "--xmom_range", type="str", default='x.x',
                  help='Range of x-momenta to consider at src/snk (default = x.x)')
parser.add_option("-y", "--ymom_range", type="str", default='x.x',
                  help='Range of y-momenta to consider at src/snk (default = x.x)')
parser.add_option("--xi", type="str", default=0,
                  help='Skewness to print momenta for (default = 0)\n If passing fraction, pass as 1.0/<denom>')
parser.add_option("-a", type="float", default=0.094,
                  help='Lattice spacing (default = 0.094)')
parser.add_option("-L", type="int", default=32,
                  help='Spatial extent (default = 32)')

(options, args) = parser.parse_args()

mass=1.1230844097
# Make an array of all computed genprop momentum transfers
Q=['0.0.1','0.0.-1','0.0.2','0.0.-2',\
   '0.1.0','0.1.1','0.1.-1','0.1.2','0.1.-2',\
   '1.0.0',\
   '1.1.0','1.1.1','1.1.-1','1.1.2','1.1.-2',\
   '2.0.1','2.0.-1','2.0.2','2.0.-2']
modQx=int(max([tx.split('.')[0] for tx in Q]))
modQy=int(max([tx.split('.')[1] for tx in Q]))
modQz=int(max([tx.split('.')[2] for tx in Q]))


# Get range of xy momenta
minXMom, maxXMom = tuple(int(x) for x in options.xmom_range.split('.'))
minYMom, maxYMom = tuple(int(y) for y in options.ymom_range.split('.'))

# Collect all the needed 2pts
twoPts=[]

# Loop over all possible ini/fin z-momenta
for pfz in range(-6,7):
    for piz in range(-6,7):

        # Loop over all possible ini/fin y-momenta
        for pfy in range(minYMom,maxYMom+1):
            for piy in range(minYMom,maxYMom+1):

                # Loop over all possible ini/fin x-momenta
                for pfx in range(minXMom,maxXMom+1):
                    for pix in range(minXMom,maxXMom+1):

                        srcMom="%s.%s.%s"%(pix,piy,piz)
                        
                        # Run over all momentum transfers
                        for q in Q:

                            # Compare against this snkMom
                            snkMom="%s.%s.%s"%(pfx,pfy,pfz)

                            # This snkMom is result of momentum conservation
                            snkMomTruth="%i.%i.%i"%(pix+int(q.split('.')[0]),
                                                    piy+int(q.split('.')[1]),
                                                    piz+int(q.split('.')[2]))

                            # Continue if snkMom != truth
                            if snkMomTruth != snkMom:
                                continue
                            else:

                                # Compute skewness/momentum transfer for these src/snkMom
                                mat=pgitd(snkMom,srcMom,mass,options.a,options.L)
                                mat.skewness()
                                mat.transfer()
                                
                                # If this computed skewness is equal to desired, then print moms
                                if mat.xi == float(eval(options.xi)):
                                    print("Skewness = %.4f for [pf, pi] = [ (%i,%i,%i), (%i,%i,%i) ]"%(float(eval(options.xi)),pfx,pfy,pfz,pix,piy,piz))
                                    twoPts.append(srcMom)
                                    twoPts.append(snkMom)


# Collect all the unique 2pts                                    
uniq2pts=[]
for t in twoPts:
    if t not in uniq2pts:
        uniq2pts.append(t)

print("Unique 2pts needed:")
for u in uniq2pts:
    print("2pt  --->  %s"%u)

                                    
# Determine all snk-momenta consistent with momentum transfer
for q in Q:

    snkMom=''
    for i in range(0,3):
        snkMom+="%d."%(int(q.split('.')[i])+int(srcMom.split('.')[i]))

    snkMom=snkMom[0:-1] # strip trailing period


    # Determine the skewness/momentum transfer
    mat=pgitd(snkMom,srcMom,mass)
    mat.skewness()
    mat.transfer()

    if not bool(options.findZeroXi):
        print("%s -- %s"%(snkMom,srcMom))
        print("   xi = %.5f  ,   t  = %.5f"%(mat.xi,mat.t))

    if bool(options.findZeroXi):
        if mat.xi == 0:
            print("Zero skewness for [pf, pi] = [ (%s,%s,%s), (%s,%s,%s) ]"\
                %(snkMom.split('.')[0],snkMom.split('.')[1],snkMom.split('.')[2],\
                  srcMom.split('.')[0],srcMom.split('.')[1],srcMom.split('.')[2]))
