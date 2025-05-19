
from cc3d.core.PySteppables import *
import numpy as np
from pathlib import Path
import os


class GrowthSteppable(SteppableBasePy): # VolumeParamSteppable inherits from SteppableBasePy Class
    def __init__(self, lambda_v, lambda_s, A_i, sigma, epsilon_P, GT, GF, dt, frequency = 1):
        SteppableBasePy.__init__(self, frequency)
        self.lambda_v = lambda_v
        self.lambda_s = lambda_s
        self.A_i = A_i
        self.sigma = sigma
        self.epsilon_P = epsilon_P
        self.GT = GT
        self.GF = GF
        self.dt = dt

    def start(self):
        for cell in self.cellList:
            cell.targetVolume = self.A_i
            cell.lambdaVolume = self.lambda_v
            cell.targetSurface = self.sigma*self.epsilon_P*np.sqrt(4*np.pi*cell.targetVolume) #formula due to circle shape of cells without contact to substrate
            cell.lambdaSurface = self.lambda_s
            cellDict=CompuCell.getPyAttrib(cell)        # dictonary for cells
            cellDict["surface_change"] = 0               # surface_change ist key in dictionary
            cellDict["age"] = 0
    
    def step(self, mcs):
        for cell in self.cellList:
            if cell.type == 1:
                cellDict = CompuCell.getPyAttrib(cell)
                cellDict["age"] += 1                    # age of every cell increases every timestep
                #print("age of cell id=", cell.id, " type:", cell.type, " = ", cellDict["age"])
                            
            # Check growth every dt-th MCS
            if (mcs%self.dt)>0:          
                #print("averaging: " + str(mcs))
                if cell.type == 1:
                    diff = cell.surface - cell.targetSurface
                    cellDict = CompuCell.getPyAttrib(cell)
                    cellDict["surface_change"] += diff   
                    #print(str(cell.id) + ': ' + str(cellDict["surface_change"]))
            
            if (mcs%self.dt)==0:         
                #print("stop averaging, surface change")
                if cell.type==1:
                    cellDict=CompuCell.getPyAttrib(cell)
                    cellDict["surface_change"] = cellDict["surface_change"]/self.dt        # calc. average surface/perimeter change over dt MCS
                    growththreshold = self.GT*cell.targetSurface
                    #print('average surface change: ', cellDict["surface_change"])
                    #print('Cell ID: ' + str(cell.id))
                    #print('MCS: ' + str(mcs))                                         
                    if (cellDict["surface_change"] > growththreshold):
                        cell.targetSurface += self.GF * cellDict["surface_change"]                   
                        cell.targetVolume = (cell.targetSurface**2)/(4 * np.pi * self.sigma**2 * self.epsilon_P**2) # due to circle shape without adhesion                      
                    cellDict["surface_change"] = 0       # surface_change wird fï¿½r alle Zellen nach dt MCS auf 0 gesetzt
                    
'''
class GrowthSteppable(SteppableBasePy): # VolumeParamSteppable inherits from SteppableBasePy Class
    def __init__(self, lambda_v, lambda_s, sigma, frequency = 1):
        SteppableBasePy.__init__(self, frequency)
        self.lambda_v = lambda_v
        self.lambda_s = lambda_s
        self.sigma = sigma

    def start(self):
        for cell in self.cellList:
            cell.targetVolume = 400
            cell.lambdaVolume = self.lambda_v
            cell.targetSurface = self.sigma * 1.225 * np.sqrt(4*np.pi*cell.targetVolume) #formula due to circle shape of cells without contact to substrate
            cell.lambdaSurface = self.lambda_s
            cellDict=CompuCell.getPyAttrib(cell)        # dictonary for cells
            cellDict["surface_change"] = 0               # surface_change ist key in dictionary
            cellDict["age"] = 0
    
    
    def step(self, mcs):
        
        growth_factor = 0.01
        growththreshold = 37
        
        for cell in self.cellList:
            if cell.type == 1:
                # Track Cell Age
                cellDict = CompuCell.getPyAttrib(cell)
                cellDict["age"] += 1                    # age of every cell increases every timestep
                #print("age of cell id=", cell.id, " type:", cell.type, " = ", cellDict["age"])
                
                # Growth based on tension dP -> w/o dt yields a more constant growth -> cell shape is still equilibrted, if the GT is high enough (relative to JCS)
                dP = cell.surface - cell.targetSurface
                if dP > growththreshold:#cell.targetSurface*0.42:                 
                    cell.targetSurface += growth_factor*dP
                    cell.targetVolume = cell.targetSurface**2/(4*np.pi* 1.225**2) # due to circle shape without adhesion
'''

class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, A_i, sigma, epsilon_P, frequency=1):
        MitosisSteppableBase.__init__(self,frequency)
        self.A_i = A_i
        self.sigma = sigma
        self.epsilon_P = epsilon_P

    def step(self, mcs):
        cells_to_divide=[]
        for cell in self.cell_list:
            if cell.type==1:
                if cell.volume>2*self.A_i:              #Mitosis Condition
                    cells_to_divide.append(cell)

        for cell in cells_to_divide:
            #print("dividing cell")
            self.divide_cell_along_minor_axis(cell)     #Definition of Mitosis type
            #print("divided cell")

    def update_attributes(self):
        parentCell=self.mitosisSteppable.parentCell
        childCell=self.mitosisSteppable.childCell
        #print("assigning atttributes")
        childCell.targetVolume = self.A_i
        childCell.targetSurface = self.sigma * self.epsilon_P * np.sqrt(4*np.pi*childCell.targetVolume)
        childCell.lambdaVolume = parentCell.lambdaVolume
        childCell.lambdaSurface = parentCell.lambdaSurface
        parentCell.targetVolume = self.A_i
        parentCell.targetSurface = self.sigma * self.epsilon_P * np.sqrt(4*np.pi*childCell.targetVolume)
        childCell.type = 1

        #print("done")
        cellDict = CompuCell.getPyAttrib(parentCell)
        cellDict["surface_change"] = 0
        cellDict["age"] = 0

        cellDict = CompuCell.getPyAttrib(childCell)
        cellDict["surface_change"] = 0
        cellDict["age"] = 0


class TissueTrackerSteppable(SteppableBasePy): # puts out data to plot the tissue (cells + surfaces)
    def __init__(self, SimID, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.SimID = SimID

        #Include SimID here to go in the right subdirectory!
        output_path = Path(__file__).parents[1] / Path('Output')
        self.set_output_dir(output_dir = output_path, abs_path=False)
        
        
    def start(self):
        output_dir = self.output_dir
        #'''
        #Output of initial condition 
        if output_dir is not None:
            # write volume pixel list to file
            output_path = Path(output_dir).joinpath('tissue_volume_' + str(0) + '.txt')
            with open(output_path, 'w') as fout:
                for cell in self.cell_list:
                    cellDict = CompuCell.getPyAttrib(cell)
                    pixel_list = self.get_cell_pixel_list(cell)
                    for pixel_tracker_data in pixel_list:
                        fout.write(str(cell.id + 1) + '\t' +
                        str(pixel_tracker_data.pixel.x) + '\t' +
                        str(pixel_tracker_data.pixel.y) + '\t' +
                        str(cell.xCOM) + '\t' +
                        str(cell.yCOM) + '\t' +
                        str(cellDict["age"]) + '\n')

            # write surface pixel list to file
            output_path = Path(output_dir).joinpath('tissue_surface_' + str(0) + '.txt')
            with open(output_path, 'w') as fout:
                for cell in self.cell_list:
                    boundary_pixel_list = self.get_cell_boundary_pixel_list(cell)
                    for boundary_pixel_tracker_data in boundary_pixel_list:
                        fout.write(str(cell.id + 1) + '\t' +
                        str(boundary_pixel_tracker_data.pixel.x) + '\t' +
                        str(boundary_pixel_tracker_data.pixel.y) + '\n')
        #'''
    def step(self, mcs):
        output_dir = self.output_dir
        if output_dir is not None:
            # write volume pixel list to file
            output_path = Path(output_dir).joinpath('tissue_volume_' + str(mcs+1) + '.txt')
            with open(output_path, 'w') as fout:
                for cell in self.cell_list:
                    cellDict = CompuCell.getPyAttrib(cell)
                    pixel_list = self.get_cell_pixel_list(cell)
                    for pixel_tracker_data in pixel_list:
                        fout.write(str(cell.id + 1) + '\t' +
                        str(pixel_tracker_data.pixel.x) + '\t' +
                        str(pixel_tracker_data.pixel.y) + '\t' +
                        str(cell.xCOM) + '\t' +
                        str(cell.yCOM) + '\t' +
                        str(cellDict["age"]) + '\n')

            # write surface pixel list to file
            output_path = Path(output_dir).joinpath('tissue_surface_' + str(mcs+1) + '.txt')
            with open(output_path, 'w') as fout:
                for cell in self.cell_list:
                    boundary_pixel_list = self.get_cell_boundary_pixel_list(cell)
                    for boundary_pixel_tracker_data in boundary_pixel_list:
                        fout.write(str(cell.id + 1) + '\t' +
                        str(boundary_pixel_tracker_data.pixel.x) + '\t' +
                        str(boundary_pixel_tracker_data.pixel.y) + '\n')


class CellTrackerSteppable(SteppableBasePy): # puts out data of cells (volume, surface, age, etc.)
    def __init__(self, SimID, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.SimID = SimID

        #Include SimID here to go in the right subdirectory!
        output_path = Path(__file__).parents[1] / Path('Output')
        self.set_output_dir(output_dir = output_path, abs_path=False)

    def start(self):
        #empty all existing cell_xxx.txt files at the start of the simulation and write header
        output_dir = self.output_dir
        for filetuple in os.walk(output_dir):
            for filelist in filetuple:
                if isinstance(filelist, list):
                    for filename in filelist:
                        if filename.startswith("cell_"):
                            output_path = Path(output_dir).joinpath(filename)
                            with open(output_path, 'w') as fout:
                                fout.write('mcs' + '\t' +
                                'type' + '\t' +
                                'volume' + '\t' +
                                'surface' + '\t' +
                                'targetVolume' + '\t' +
                                'targetSurface' + '\n')
        #Output of initial condition                      
        for cell in self.cell_list:
            output_path = Path(output_dir).joinpath('cell_' + str(cell.id + 1) + '.txt')
            with open(output_path, 'a') as fout: 
                fout.write(str(0) + '\t' +
                str(cell.type) + '\t' +
                str(cell.volume) + '\t' +
                str(cell.surface) + '\t' +
                str(cell.targetVolume) + '\t' +
                str(cell.targetSurface) + '\n')

    def step(self, mcs):
        output_dir = self.output_dir
        for cell in self.cell_list:
            output_path = Path(output_dir).joinpath('cell_' + str(cell.id + 1) + '.txt')
            with open(output_path, 'a') as fout: #problem: file doesn't get overwriten at the start of the simulation (delete files )
                fout.write(str(mcs+1) + '\t' +
                str(cell.type) + '\t' +
                str(cell.volume) + '\t' +
                str(cell.surface) + '\t' +
                str(cell.targetVolume) + '\t' +
                str(cell.targetSurface) + '\n')

class MacroscopicVolumeTrackerSteppable(SteppableBasePy): # puts out data to plot the volume of the tissue (sum of all cell volumes)
    def __init__(self, SimID, frequency=50):
        SteppableBasePy.__init__(self, frequency)
        self.SimID = SimID

        output_path = Path(__file__).parents[1] / Path('Output')
        self.set_output_dir(output_dir = output_path, abs_path=False)  
    
    def start(self):
        #empty the existing MacroscopicVolume.txt file at the start of the simulation and write header
        output_dir = self.output_dir
        for filetuple in os.walk(output_dir):
            for filelist in filetuple:
                if isinstance(filelist, list):
                    for filename in filelist:
                        if filename.startswith("MacroscopicVolume"):
                            output_path = Path(output_dir).joinpath(filename)
                            with open(output_path, 'w') as fout:
                                fout.write('mcs' + '\t' + 'volume' + '\n')
                                
        #Output of initial condition                             
        MacroVol = 0
        for cell in self.cell_list: #get the sum of all cell volumes as macroscopic tissue volume -> This is the projected Tissue Area in 2D 
            if cell.type==1:
                MacroVol += cell.volume
                    
        output_path = Path(output_dir).joinpath('MacroscopicVolume' + '.txt')
        with open(output_path, 'a') as fout:
            fout.write(str(0) + '\t' + str(MacroVol) + '\n')           
     
    def step(self, mcs):
        output_dir = self.output_dir
        MacroVol = 0
        for cell in self.cell_list: #get the sum of all cell volumes as macroscopic tissue volume -> This is the projected Tissue Area in 2D 
            if cell.type==1:
                MacroVol += cell.volume
                    
        output_path = Path(output_dir).joinpath('MacroscopicVolume' + '.txt')
        with open(output_path, 'a') as fout:
            fout.write(str(mcs+1) + '\t' + str(MacroVol) + '\n')