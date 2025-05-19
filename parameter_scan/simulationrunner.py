import numpy as np
import os
from pathlib import Path
import pandas as pd


class simulationrunner:
    def __init__(self, root_dir, scenario_name, parameter_scan_name, CC3Ddir):
        self.root_dir = root_dir
        self.scenario_name = scenario_name
        self.parameter_scan_name = parameter_scan_name
        self.CC3Ddir = CC3Ddir

    def check_and_repeat_simulation(self,SimID):
        ### repeat simulations, where there is no cell 1!!!
        self.outputdata_dir = self.root_dir / Path('scenarios') / self.scenario_name / self.parameter_scan_name / Path(str(SimID) + "_Simulation") / self.scenario_name / Path('Output')
        cell_1_path = os.path.join(self.outputdata_dir, 'cell_1.txt')
        if os.path.isfile(cell_1_path):
            return
        else:
            print("cell_1.txt does not exist. Repeat simulation...")
            print()
            os.chdir(self.CC3Ddir) #goes to the CC3D Directory to run CompuCell from there
            cc3dfile_path = self.root_dir / Path('scenarios') / self.scenario_name / self.parameter_scan_name / Path(str(SimID) + "_Simulation") / self.scenario_name / Path(self.scenario_name + '.cc3d')
            print(cc3dfile_path)
            #use the for-loop to iterate over the differnet simulations:
            command = "runScript.bat -i " + str(cc3dfile_path)
            os.system(command) #runs the simulation