import parameter_scan.parameterscan as scan

from pathlib import Path
import numpy as np


root_dir =  Path('C:/Users/lenna/Documents/GitHub/tension-driven-cpm') #set this to the root of the repository
senario_name="Monolayer"
parameter_scan_name = 'parameter_scan_GT'
CC3Ddir = "C:/CompuCell3D"  # define path where CC3D is installed!
Scan =  scan.parameter_scan(root_dir, senario_name, parameter_scan_name, CC3Ddir)

################################################################################
#definition of parameter data
# Predefined values
#tau scan
#values = [10, 50, 100, 150, 200] 

# GF scan
#values = [1, 0.5, 0.1, 0.02, 0.01, 0.00625, 0.005, 0.002] 

# GT scan
values = [1, 0.5, 0.25, 0.05, 0] 

# Create a list where each value is repeated 10 times
repeated_values = [val for val in values for _ in range(10)]
N = len(repeated_values)

data = []
SimID = 1
for i in range(N):
    data.append([
        SimID, '10', '300', '100', '1', '1', '1', '50', '0', '-50', '15001', '400', '1', '1.273', repeated_values[i], '0.005', '10'
    ])
    SimID += 1
################################################################################

Scan.WriteParameterScanFile(data)
Scan.CreateSimulations()
Scan.ExecuteSimulations()