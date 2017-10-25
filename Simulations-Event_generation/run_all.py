# Script to run 'run_MG5_pythia_jet_img.py', which runs MG5 and Pythia and returns tuples of 4-momentum vectors for jets and subjets

#---------------------------------------------------------------------------------------------

import os

#---------------------------------------------------------------------------------------------
#  GLOBAL VARIABLES

signal='tt'
background='QCD_dijet'

#Number of signal and background jobs to send to the hexfarm cluster
Nruns_signal=str(60)
Nruns_background=str(120)

sleep_time=str(5)

mypath ='/het/p1/macaluso/deep_learning/'
# MG5_process_folders = ['tt','QCD_dijet']

#------------------------------------------
# MG5 input values
MG5_EVENTS=str(10000)
MG5_PTJ1MIN=str(325)#Min pT required in MG5 for the leading jet in pT
MG5_PTJ1MAX=str(400)
MG5_ETAMAX=str(2.5)

#------------------------------------------
# PYTHIA input values
jet_eta_max=str(2.5)
R_jet=str(1.5)
pTjetMin=str(350) #Min pT for all jets


#---------------------------------------------------------------------------------------------
# Folders with executables .sh and .jdl files for job submission 

paths_signal = [mypath+'exec_'+signal+'_'+MG5_EVENTS+'_'+pTjetMin+'_'+R_jet]
os.system("mkdir -p "+paths_signal[0])

paths_background = [mypath+'exec_'+background+'_'+MG5_EVENTS+'_'+pTjetMin+'_'+R_jet]
os.system("mkdir -p "+paths_background[0])

#---------------------------------------------------------------------------------------------
# Generate exc files

os.chdir(mypath)
os.system('python run_MG5_pythia_jet_img.py '+signal+' '+MG5_EVENTS+' '+MG5_PTJ1MIN+' '+MG5_PTJ1MAX+' '+MG5_ETAMAX+' '
        +jet_eta_max+' '+R_jet+' '+pTjetMin+' '+Nruns_signal)

os.system('python run_MG5_pythia_jet_img.py '+background+' '+MG5_EVENTS+' '+MG5_PTJ1MIN+' '+MG5_PTJ1MAX+' '+MG5_ETAMAX+' '
        +jet_eta_max+' '+R_jet+' '+pTjetMin+' '+Nruns_background)
print('Generating exec f.sh and .jdl files')


#---------------------------------------------------------------------------------------------
# Submit jobs to the hexfarm cluster

for path in paths_signal:
    os.chdir(path)
    print('Sending signal jobs to the cluster')
    os.system("ls")
    with open("do_all.src","r") as f: 
        for line in f.readlines():
            os.system(line.strip())
            os.system('sleep '+sleep_time+'\n')


for path in paths_background:
    os.chdir(path)
    print('Sending background jobs to the cluster')
    os.system("ls")
    with open("do_all.src","r") as f: 
        for line in f.readlines():
            os.system(line.strip())
            os.system('sleep '+sleep_time+'\n')

