import numpy as np
import os, shutil
from pathlib import Path
import sys
import pandas as pd

class parameter_scan:
    def __init__(self, root_dir, scenario_name, parameter_scan_name, CC3Ddir):
        self.root_dir = root_dir
        self.scenario_name = scenario_name
        self.parameter_scan_name = parameter_scan_name
        self.CC3Ddir = CC3Ddir

    def WriteParameterScanFile(self, data):
        #define the simulations directory where all simulations and the parameter scan file are stored
        self.simdir = self.root_dir / 'scenarios' / self.scenario_name / self.parameter_scan_name
        self.simdir.mkdir(parents=True, exist_ok=True)

        filename = self.scenario_name + "ParameterScan.xlsx"  # File to store the parameters for all simulations
        self.ParameterScanFilePath = self.simdir / filename

        df = pd.DataFrame(data, columns=['SimID', 'NO', 'xdim', 'ydim', 'Temp', 'lambda_v', 'lambda_s', 'J10', 'J11', 'J12', 'steps', 'A_i', 'sigma', 'epsilon_P', 'GT', 'GF', 'dt'])
        df.to_excel(self.ParameterScanFilePath, index=False)

    def CreateSimulations(self):
        #function to search and replace a string in a file
        def search_replace(target, replacement, filename):
            with open(filename, 'r') as file :
                filedata = file.read()
            # Replace the target string
            filedata = filedata.replace(target, replacement)
            # Write the file out again
            with open(filename, 'w') as file:
                file.write(filedata)
        
        # read ParameterScanFile.xlsx and create the corresponding subdirectories for all simulations (based on SimID)
        df = pd.read_excel(self.ParameterScanFilePath)
        self.N_Simulations = df.shape[0]

        for i in range(0,self.N_Simulations):
            simulation_parameter = df.loc[i]
            #store the paramters in the right vairables
            SimID = int(simulation_parameter[0])
            NO = int(simulation_parameter[1])
            xdim = int(simulation_parameter[2])
            ydim = int(simulation_parameter[3])
            Temp = simulation_parameter[4]
            lambda_v = simulation_parameter[5]
            lambda_s = simulation_parameter[6]
            J10 = simulation_parameter[7]
            J11 = simulation_parameter[8]
            J12 = simulation_parameter[9]
            steps = int(simulation_parameter[10])
            A_i = int(simulation_parameter[11])
            sigma = simulation_parameter[12]
            epsilon_P = simulation_parameter[13]
            GT = simulation_parameter[14]
            GF = simulation_parameter[15]
            dt = simulation_parameter[16]
            ####################################################
            #create subdirectories for the different simulations
            subdir_path = self.simdir / Path(str(SimID) + "_Simulation")
            subdir_path.mkdir(parents=True, exist_ok=True)
        
            ####################################################
            #copy template-simulation into the subdirectory
            # Setting the source and the destination folders
            src = self.root_dir / 'scenarios' / self.scenario_name / Path("Template" + self.scenario_name) / Path(self.scenario_name)
            dst = self.simdir / Path(str(SimID) + "_Simulation") / Path(self.scenario_name)

            #delete the destination directory if it already exists (code can run multiple times without errors)
            if os.path.exists(dst):
                shutil.rmtree(dst) 
            #copy folder from sorce to destination
            shutil.copytree(src, dst)

            ####################################################
            #write the SimulationParameter.xlsx file
            param_data = {
                'SimID': [SimID],
                'NO': [NO],
                'xdim': [xdim],
                'ydim': [ydim],
                'Temp': [Temp],
                'lambda_v': [lambda_v],
                'lambda_s': [lambda_s],
                'J10': [J10],
                'J11': [J11],
                'J12': [J12],
                'steps': [steps],
                'A_i': [A_i],
                'sigma': [sigma],
                'epsilon_P': [epsilon_P],
                'GT': [GT],
                'GF': [GF],
                'dt': [dt]
            }
            df_temp = pd.DataFrame(param_data) # Create a DataFrame from the dictionary
            dst_parameterfile = dst / Path(str(SimID) + '_SimulationParameters.xlsx')   # Define the destination Excel file path
            df_temp.to_excel(dst_parameterfile, index=False) # Write the DataFrame to an Excel file

            ####################################################
            #update the simulation parameter in the python file
            dst_pythonfile = dst / Path('Simulation') / Path(self.scenario_name + '.py')

            #search and replace the parameter in AutomationTissueGrowth.py
            search_replace('SimID = ?', 'SimID = ' + str(SimID), dst_pythonfile)
            search_replace('NO = ?', 'NO = ' + str(NO), dst_pythonfile)
            search_replace('xdim = ?', 'xdim = ' + str(xdim), dst_pythonfile)
            search_replace('ydim = ?', 'ydim = ' + str(ydim), dst_pythonfile)
            search_replace('Temp = ?', 'Temp = ' + str(Temp), dst_pythonfile)
            search_replace('lambda_v = ?', 'lambda_v = ' + str(lambda_v), dst_pythonfile)
            search_replace('lambda_s = ?', 'lambda_s = ' + str(lambda_s), dst_pythonfile)
            search_replace('J10 = ?', 'J10 = ' + str(J10), dst_pythonfile)
            search_replace('J11 = ?', 'J11 = ' + str(J11), dst_pythonfile)
            search_replace('J12 = ?', 'J12 = ' + str(J12), dst_pythonfile)
            search_replace('steps = ?', 'steps = ' + str(steps), dst_pythonfile)
            search_replace('A_i = ?', 'A_i = ' + str(A_i), dst_pythonfile)
            search_replace('sigma = ?', 'sigma = ' + str(sigma), dst_pythonfile)
            search_replace('epsilon_P = ?', 'epsilon_P = ' + str(epsilon_P), dst_pythonfile)
            search_replace('GT = ?', 'GT = ' + str(GT), dst_pythonfile)
            search_replace('GF = ?', 'GF = ' + str(GF), dst_pythonfile)
            search_replace('dt = ?', 'dt = ' + str(dt), dst_pythonfile)

        
    def ExecuteSimulations(self):
        os.chdir(self.CC3Ddir) #goes to the CC3D Directory to run CompuCell from there
        for i in range(1, self.N_Simulations+1):
            cc3dfile_path = self.root_dir / Path('scenarios') / self.scenario_name / self.parameter_scan_name / Path(str(i) + "_Simulation") / self.scenario_name / Path(self.scenario_name + '.cc3d')
            #use the for-loop to iterate over the differnet simulations:
            command = "runScript.bat -i " + str(cc3dfile_path)
            os.system(command) #runs the simulation