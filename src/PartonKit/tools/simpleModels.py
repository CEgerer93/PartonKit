#!/usr/bin/python3

import numpy as np
import scipy.special as sp
import scipy.integrate as integrate
import sys,optparse
import matplotlib.pyplot as plt

sys.path.append('/home/colin/QCD/pseudoDists/strFuncViz')
import pdf_utils as pdf


def hyp_0F1(n,arg,trunc=100):
    hyp = 0.0
    for k in range(0,trunc+1):
        hyp += (1.0*np.power(arg,k))/(sp.gamma(n+k)*sp.factorial(k))
    return hyp
    


def func(nus,xi,a,b,N):
    forwardPDF = pdf.pdf2('2-param pdf', 'C', 'red',\
                          {'alpha': [a], 'beta': [b], 'norm': None}, False, 1)
    forwardPDF.parse()


    # Final result
    res = []

    for nu in nus:
        real = lambda beta : np.cos(nu*beta)*forwardPDF.pdf(beta)*\
            hyp_0F1(N+1.5,-0.25*nu**2*xi**2*np.power(1-abs(beta),2))*sp.gamma(N+1.5);
        imag = lambda beta : np.sin(nu*beta)*forwardPDF.pdf(beta)*\
            hyp_0F1(N+1.5,-0.25*nu**2*xi**2*np.power(1-abs(beta),2))*sp.gamma(N+1.5);


        res.append(complex(integrate.quad(real,0,1)[0],integrate.quad(imag,0,1)[0]))


    return res


# Parse command line options
usage = "usage: %prog [options] "
parser = optparse.OptionParser(usage) ;

parser.add_option("-a", "--pdfAlpha", type="float", default=-0.5,
                  help="alpha of x^\alpha in 2-param pdf form (default = '-0.5')")
parser.add_option("-b", "--pdfBeta", type="float", default=3.0,
                  help="beta of (1-x)^\beta in 2-param pdf form is z-momenum (default='3.0')")

(options, args) = parser.parse_args()


nus=np.linspace(0,5,100)



for xi in [0,1,-1,1.0/3,-1.0/3]:
    for N in [0,1,2,3]:
        plt.plot(nus,func(nus,xi,options.pdfAlpha,options.pdfBeta,N),\
                 label="Xi = %i; N = %i"%(xi,N))


plt.legend()
plt.show()






   
