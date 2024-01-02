#!/usr/bin/python3

import matplotlib.pyplot as plt

# Initialize figure properties
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}') # for mathfrak
plt.rcParams["mathtext.fontset"]="stix"
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 20 # 15 (for '/g_A(\mu^2)')
plt.rcParams['errorbar.capsize'] = 2.0
# plt.rcParams['ytick.direction'] = 'out'
plt.rc('xtick.major',size=10)
plt.rc('ytick.major',size=10)
plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)
plt.rc('axes',labelsize=24)
truthTransparent=False
LegendFrameAlpha=1.0 #0.8
legendFaceColor="white"
suffix=''
form='pdf'
# # Optionally swap default black labels for white
# if options.lightBkgd == 0:
#     truthTransparent=True
#     plt.rcParams['text.color'] = 'white'
#     plt.rcParams['axes.edgecolor'] = 'white'
#     # plt.rcParams['legend.edgecolor'] = "#1b212c"
#     plt.rc('axes',edgecolor='white')
#     plt.rc('axes',labelcolor='white')
#     plt.rc('xtick',color='white')
#     plt.rc('ytick',color='white')
#     plt.rc('text',color='white')
#     LegendFrameAlpha=0.0
#     legendFaceColor="#1b212c"
#     suffix='.dark'
#     form='png'
