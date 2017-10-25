#/usr/bin/env python
# Script to create bash files to run Madgraph

import sys, os, time, fileinput

#To run: Run this script with run_all.py
# python run_MG5_pythia_jet_img.py '+signal+' '+MG5_EVENTS+' '+MG5_PTJ1MIN+' '+MG5_PTJ1MAX+' '+MG5_ETAMAX+' '+jet_eta_max+' '+R_jet+' '+pTjetMin+' '+Nruns_signal

#---------------------------------------------------------------------------------------------
#GLOBAL VARIABLES

# The MG5_process folder has the entry for the directory with the unweightes_events.lhe from MG5
MG5_process=sys.argv[1]
MYEVENTS=sys.argv[2]
MYPTJ1MIN=sys.argv[3]#Min pT required in MG5 for the leading jet in pT
MYPTJ1MAX=sys.argv[4]
MYETAMAX=sys.argv[5]

eta_max=sys.argv[6]
Rjet=sys.argv[7]
pTjetMin=sys.argv[8] #Min pT for all jets

Nruns=int(sys.argv[9]) #Number of MG5 runs that we will do (we update the seed number for each of them)

Nevents=MYEVENTS
# Nevents=str(4)

print('jet1pT_min={}'.format(MYPTJ1MIN))
print('Nevents = {}'.format(Nevents))

mainpath="/het/p1/macaluso/"
username="macaluso"
main_dir=mainpath+'deep_learning/'

MG5="MG5_aMC_v2.4.3.tar.gz" 
MG5_dir='MG5_aMC_v2_4_3/'
MGTAR="/het/p1/"+username+"/"+MG5+"" 

MADGRAPH_DIRECTORY="/het/p1/"+username+'/'+MG5_dir

pathPythia= os.path.join(mainpath,'pythia8219/examples/')

jets_filename='jets'
subjets_filename='subjets'

#---------------------------------------------------------------------------------------------
#generate command for Madgraph
gen_command = "/cms/base/python-2.7.1/bin/python ./bin/generate_events 0 "

#initialization for random number seed
seed = 0
# Nruns=90 #Number of MG5 runs that we will do (we update the seed number for each of them)

#make folder for bash scripts
executedir = main_dir+'exec_'+MG5_process+'_'+MYEVENTS+'_'+pTjetMin+'_'+Rjet
#os.system("rm -rf "+executedir)
os.system("mkdir -p "+executedir)

#directory to collect lhe files results 
LHE_RESULTS_DIR =main_dir+'LHE/results_'+MG5_process+'_'+MYEVENTS+'_'+pTjetMin+'_'+Rjet
#make directory if it doesn't already exist
os.system("mkdir -p "+LHE_RESULTS_DIR)

#Directory to collect results in after PYTHIA
RESULTS_DIR = main_dir+'results_'+MG5_process+'_'+MYEVENTS+'_'+pTjetMin+'_'+Rjet
os.system("mkdir -p "+RESULTS_DIR)
   
#open file to contain submission commands for jdl files
dofile = open(executedir+"/do_all.src",'w')

#---------------------------------------------------------------------------------------------
#////////////////////////////////////////////////////////////////////////////////////////////
#---------------------------------------------------------------------------------------------
# Loop that copies MG5 to the machine in the cluster, updates the random seed number, runs MG5 and saves the unweighted_events.lhe
# Then runs PYTHIA 8 and saves the .npy Numpy arrays of 4-momentum vectors 
for i in range(0,Nruns):

   seed += 1
   name = MG5_process+'_'+str(seed)

   #define name of template copy and specifc run name
   
   RUN_NAME = name
   
   localdir = "$_CONDOR_SCRATCH_DIR"
   
   #---------------------------------------------------------------------------------------------
   #write out python script for job execution
   execfilename = "exec_"+name+".sh"
   
   executefile = open(executedir+"/"+execfilename,'w')
   executefile.write("#!/bin/bash\n")
   executefile.write("export VO_CMS_SW_DIR=\"/cvmfs/cms.cern.ch\"\n")
   executefile.write("export COIN_FULL_INDIRECT_RENDERING=1\n")
   executefile.write("export SCRAM_ARCH=\"slc6_amd64_gcc481\"\n")

   #---------------------------------------------------------------------------------------------
   # Copy MG5 to the machine in the cluster
   executefile.write("cp -r "+MGTAR+" "+localdir+"\n") 
   executefile.write("cd "+localdir+"\n")
   executefile.write("tar -xzvf "+MG5+"\n")
   #executefile.write("cd MG5_aMC_v2_3_3/pythia-pgs \n")
   #executefile.write("make \n")
   #executefile.write("cd ../Delphes \n")
   #executefile.write("make \n")
   #executefile.write("cd .. \n")

   #---------------------------------------------------------------------------------------------
   #copy template directory to new location, and update its random number seed and run name
   executefile.write("cp -r "+MADGRAPH_DIRECTORY+MG5_process+".tar.gz "+localdir+"/"+MG5_dir+MG5_process+".tar.gz \n")
   executefile.write("cd "+MG5_dir+" \n")
   executefile.write("tar -xzvf "+MG5_process+".tar.gz \n")
   executefile.write("mv "+MG5_process+" "+RUN_NAME+"\n")
   
   #---------------------------------------------------------------------------------------------
   #Update values in the run_card.dat
   run_card = localdir+"/"+MG5_dir+RUN_NAME+"/Cards/run_card.dat"
   executefile.write("sed -i \'s/MYSEED/"+str(seed)+"/\' "+run_card+"\n")
   executefile.write("sed -i \'s/RUNNAME/"+RUN_NAME+"/\' "+run_card+"\n")
   executefile.write("sed -i \'s/MYEVENTS/"+MYEVENTS+"/\' "+run_card+"\n")
   executefile.write("sed -i \'s/MYPTJ1MIN/"+MYPTJ1MIN+"/\' "+run_card+"\n")
   executefile.write("sed -i \'s/MYPTJ1MAX/"+MYPTJ1MAX+"/\' "+run_card+"\n")
   executefile.write("sed -i \'s/MYETAMAX/"+str(MYETAMAX)+"/\' "+run_card+"\n")

   #---------------------------------------------------------------------------------------------
   #Run MG5  
   executefile.write("cd "+localdir+"/"+MG5_dir+RUN_NAME+" \n")
   executefile.write(gen_command+RUN_NAME+"\n")
   
   #---------------------------------------------------------------------------------------------
   #Save the output files
   executefile.write("cp "+localdir+"/"+MG5_dir+RUN_NAME+"/Events/"+RUN_NAME+"/unweighted_events.lhe.gz "+LHE_RESULTS_DIR +"/"+RUN_NAME+"_unweighted.lhe.gz \n")
   executefile.write("cp "+localdir+"/"+MG5_dir+RUN_NAME+"/Events/"+RUN_NAME+"/*banner.txt "+LHE_RESULTS_DIR +"/"+RUN_NAME+"_banner.txt\n")

   #---------------------------------------------------------------------------------------------
   #Copy lhe file to local dir
   executefile.write("mv "+localdir+"/"+MG5_dir+RUN_NAME+"/Events/"+RUN_NAME+"/unweighted_events.lhe.gz "+localdir+"/"+name+'_unweighted_events.lhe.gz \n')
   executefile.write("cd "+localdir+"\n")
   executefile.write("gunzip "+name+'_unweighted_events.lhe.gz \n')
           
   #Stable-------------------------------------------------------------
   #///////////////////////////////////////PYTHIA//////////////////////////////////
   #Stable-------------------------------------------------------------

   # runs Pythia 8 with stable chi0, RPV or HV decays specified in the cmnd files
   executefile.write("cp -r "+pathPythia+'Makefile.inc '+localdir+"/"+"Makefile.inc \n")
   executefile.write('python '+pathPythia+'pythia_LHE_image.py '+name+'_unweighted_events.lhe '+Nevents+' '
      +jets_filename+' '+subjets_filename+' '+eta_max+' '+Rjet+' '+pTjetMin+' > '+RUN_NAME+'_pythia.out \n')
   
   executefile.write("cp "+RUN_NAME+"_pythia.out "+RESULTS_DIR+"/"+RUN_NAME+"_pythia.out\n")
   executefile.write("cp "+jets_filename+".npy "+RESULTS_DIR+"/"+RUN_NAME+'_'+jets_filename+'.npy \n')
   executefile.write("cp "+subjets_filename+".npy "+RESULTS_DIR+"/"+RUN_NAME+'_'+subjets_filename+'.npy \n')


   executefile.close()
   
   os.system("chmod u+x "+executedir+"/"+execfilename)
   
   #---------------------------------------------------------------------------------------------
   #write out jdl script for job submission
   jdlfilename = "exec_"+name+".jdl.base"
   jdlfile = open(executedir+"/"+jdlfilename,'w')
   jdlfile.write("universe = vanilla\n")
   jdlfile.write("+AccountingGroup = \"group_rutgers."+username+"\"\n")
   jdlfile.write("Executable ="+executedir+"/"+execfilename+"\n")
   jdlfile.write("getenv = True\n")
   jdlfile.write("should_transfer_files = NO\n")
   jdlfile.write("Arguments = \n")
   jdlfile.write("Output = /het/p1/"+username+"/condor/"+RUN_NAME+".out\n")
   jdlfile.write("Error = /het/p1/"+username+"/condor/"+RUN_NAME+".err\n")
   jdlfile.write("Log = /het/p1/"+username+"/condor/"+RUN_NAME+".condor\n")
   jdlfile.write("Queue 1\n")
   jdlfile.close()

   #dofile.write("sleep 2 \n")
   
   dofile.write("condor_submit "+jdlfilename+"\n")
   

dofile.close()

