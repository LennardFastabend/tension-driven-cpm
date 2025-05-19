from pathlib import Path
import numpy as np
import pandas as pd

class cell:
    def __init__(self, id = None, type = None, target_volume = None, volume = None, volume_pixel = None, target_surface = None, surface = None, surface_pixel = None, age = None):
        self.id = id
        self.type = type
        self.target_volume = target_volume
        self.volume = volume
        self.volume_pixel = volume_pixel
        self.target_surface = target_surface
        self.surface = surface
        self.surface_pixel = surface_pixel
        self.age = age

class configuration:
    def __init__(self, mcs, simid, datadir):
        self.simid = simid
        self.mcs = mcs
        self.datadir = datadir
        self.cell_list = self.generate_cell_list()
        self.par = self.read_simulation_parameter()
        self.pixel_grid = np.zeros((self.par["xdim"],self.par["ydim"]))

        #fill the grid with coresponding cell id's from the cell_list
        for cell in self.cell_list:
            for i in range(cell.volume):
                #cell.volume_pixel[i][x=0 or y=1]
                x_position = cell.volume_pixel[i][0]
                y_position = cell.volume_pixel[i][1]
                self.pixel_grid[x_position][y_position] = cell.id
        print('Configuration of MCS:', self.mcs, 'initialised!' )

    def generate_cell_list(self):
        #create cell_list from simulation data at the given mcs
        filepath_volume_pixel = self.datadir / Path('tissue_volume_' + str(self.mcs) + '.txt')
        filepath_surface_pixel = self.datadir / Path('tissue_surface_' + str(self.mcs) + '.txt')

        with open(filepath_volume_pixel) as data:
            volume_pixel_data = np.genfromtxt(data, dtype='int', delimiter='\t')
        with open(filepath_surface_pixel) as data:
            surface_pixel_data = np.genfromtxt(data, dtype='int', delimiter='\t')

        '''
        ..._pixel_data[:,0] is cell.id
        ..._pixel_data[:,1] is x coordinate of pixel
        ..._pixel_data[:,2] is y coordinate of pixel
        ...(further output)
        '''
        #create a cell_list and add cell objects
        number_cells = max(volume_pixel_data[:,0])
        cell_list = []
        for i in range(1, number_cells+1):
            global cell
            temp_cell = cell()
            temp_cell.id = i

            #read in data of cell with cell.id = i
            filepath_cell_info = self.datadir / Path('cell_' + str(temp_cell.id) + '.txt')
            with open(filepath_cell_info) as data:
                cell_info_data = np.genfromtxt(data, dtype='int', delimiter='\t')
            if cell_info_data.ndim == 1: #this enshures, that cell_info_data is a 2D array (even if it has only one line)
                cell_info_data = cell_info_data.reshape(1, -1)

            #assign values for the properties of certain cell at the given mcs
            timepoints = list(cell_info_data[:,0])
            if self.mcs in timepoints: #if condition secures that only cells are added to the cell list that are "alive/present" at the given mcs
                row_index = timepoints.index(self.mcs)

                temp_cell.type = cell_info_data[row_index, 1]
                temp_cell.volume = cell_info_data[row_index, 2]
                temp_cell.surface = cell_info_data[row_index, 3]
                temp_cell.target_volume = cell_info_data[row_index, 4]
                temp_cell.target_surface = cell_info_data[row_index, 5]

                #add volume and surface pixel to cell
                #only take pixels that belong to cell.id!
                temp_cell.volume_pixel = []
                for volume_pixel_data_row in volume_pixel_data:
                    if volume_pixel_data_row[0] == temp_cell.id:
                        temp_cell.volume_pixel.append(volume_pixel_data_row[1:3])
                        temp_cell.age = volume_pixel_data_row[5]

                temp_cell.surface_pixel = []
                for surface_pixel_data_row in surface_pixel_data:
                    if surface_pixel_data_row[0] == temp_cell.id:
                        temp_cell.surface_pixel.append(surface_pixel_data_row[1:3])

                #add cell to cell_list
                cell_list.append(temp_cell)
        return cell_list

    def read_simulation_parameter(self):
        #read parameters from SimulationParameters.xlsx
        file_path = self.datadir.parent / Path(str(self.simid)+'_SimulationParameters.xlsx')
        # Read the Excel file
        df = pd.read_excel(file_path)
        simulation_parameter = df.loc[0]
    
        SimID = int(simulation_parameter[0])
        NO = int(simulation_parameter[1])
        xdim = int(simulation_parameter[2])
        ydim = int(simulation_parameter[3])
        lambda_v = simulation_parameter[4]
        lambda_s = simulation_parameter[5]
        J10 = simulation_parameter[6]
        J11 = simulation_parameter[7]
        J12 = simulation_parameter[8]
        
        simulation_parameter_dict = {"SimID":SimID,"NO":NO,"xdim":xdim,"ydim":ydim,"lambda_v":lambda_v,"lambda_s":lambda_s,"J10":J10,"J11":J11,"J12":J12}
        return simulation_parameter_dict
