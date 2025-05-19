import data_analysis.CPM_classes as CPM
import data_analysis.visualizer as artist

import data_analysis.CPM_classes as CPM
import data_analysis.visualizer as artist
import parameter_scan.simulationrunner as runner
from data_analysis.analyzer import *

from pathlib import Path
import sys
import numpy as np

CC3Ddir = "C:/CompuCell3D"
root_dir =  Path('C:/Users/lenna/Documents/GitHub/tension-driven-cpm') #set this to the root of the repository
scenario_name='Monolayer'
parameter_scan_name='parameter_scan_GT'

### Check and repeat simulations if the output is incomplete
# This is sometimes needed, when the output is labeled incorrect due to unknown reasons
'''
#The sim checker repeats simulations, when some outputfiles are missing! (Happens randomly for some simulations without a known reason)
for simid in np.arange(1,2):
    print('Check SimID: ', simid)
    sim_checker = runner.simulationrunner(root_dir, scenario_name, parameter_scan_name, CC3Ddir)
    sim_checker.check_and_repeat_simulation(simid)
#'''

### Create Configuration Images ###
#'''
for simid in np.arange(1,50,10):
    for mcs in np.arange(10001,10002,1000):
        print(simid, mcs)
        datadir = Path('scenarios/'+scenario_name+'/'+parameter_scan_name+'/'+str(simid)+'_Simulation/'+scenario_name+'/'+'Output')

        outputdir = Path('scenarios/'+scenario_name+'/'+parameter_scan_name+'/Images_pdf')
        outputdir.mkdir(parents=True, exist_ok=True)

        config = CPM.configuration(mcs, simid, datadir)
        image_vis = artist.visualizer(outputdir=outputdir, show_image=False, save_image=True)
        image_vis.visualiseConfiguration(config, 'Time: ' + str(mcs))#, textstr='MCS: ' + str(mcs))
#'''



#sys.exit()




#'''
### Plot Data ###

SimulationsDir = root_dir / Path("scenarios") / Path(scenario_name) / Path(parameter_scan_name)
ResultDir = SimulationsDir / Path('Results')
ResultDir.mkdir(parents=True, exist_ok=True)

List_of_SimID_lists = []

### Plot GrowthCurves for given SimIds ###
# gernerate time plots of the tissue volume development for diffent simulations
'''
Condition1_SimID_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Condition2_SimID_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Condition3_SimID_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
Condition4_SimID_list = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
Condition5_SimID_list = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
Condition6_SimID_list = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
Condition7_SimID_list = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
Condition8_SimID_list = [71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
#'''

#'''
Condition1_SimID_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Condition2_SimID_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Condition3_SimID_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
Condition4_SimID_list = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
Condition5_SimID_list = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
#'''
#Condition1_SimID_list = [1]
#Condition2_SimID_list = [2]
#Condition3_SimID_list = [3]
#Condition4_SimID_list = [4]
#Condition5_SimID_list = [5]
#Condition6_SimID_list = [6]
#Condition7_SimID_list = [7]

List_of_SimID_lists.append(Condition5_SimID_list)
List_of_SimID_lists.append(Condition4_SimID_list)
List_of_SimID_lists.append(Condition3_SimID_list)
List_of_SimID_lists.append(Condition2_SimID_list)
List_of_SimID_lists.append(Condition1_SimID_list)
#List_of_SimID_lists.append(Condition6_SimID_list)
#List_of_SimID_lists.append(Condition7_SimID_list)
#List_of_SimID_lists.append(Condition8_SimID_list)





#tau scan
#labels = [f"$\\tau={value}$" for value in np.array([10, 50, 100, 150, 200])]
#title = f"Influence of $\\tau$"
#parameter_label = r"$k=0.1$" + "\n" + r"$G_{th}=0.005$"

# GF scan
#labels = [f"$k={value}$" for value in np.array([1, 0.5, 0.1, 0.02, 0.01, 0.00625, 0.005, 0.002])]
#colors = ['#17becf', '#bcbd22', '#1f77b4', 'green', 'orange', 'red', 'purple', '#e377c2']
#title = f"Influence of $k$"
#parameter_label = r"$\tau=10$" + "\n" + r"$G_{th}=0.005$


# GT scan
labels = [f"$G_{{{'th'}}}={value}$" for value in np.array([0, 0.05, 0.25, 0.5, 1])] #GT labels
title = f"Influence of $G_{{{'th'}}}$"
parameter_label = r"$\tau=10$" + "\n" + r"$k=0.005$"
colors = ['#1a759f', 'purple', '#f3722c', '#43aa8b', '#ff4d6d']


#labels = [rf"$\alpha={value}$" for value in np.array([0, 15, 30, 45, 60, 75, 90])] #GT labels
#title = f"Serrated Edges"
#parameter_label = f"$\\tau=10$, $k=0.005$"
#colors = ['#1a759f', 'purple', '#f3722c', '#43aa8b', '#ff4d6d']


curve_vis = artist.visualizer(outputdir=ResultDir, show_image=False, save_image=True)
curve_vis.visualiseMeanGrowthCurves(List_of_SimID_lists, labels, SimulationsDir, scenario_name, title, parameter_label, colors=colors)

#'''