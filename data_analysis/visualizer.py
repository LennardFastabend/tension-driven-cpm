import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os, re


import cv2
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

class visualizer:
    def __init__(self, outputdir, show_image=True, save_image=False):
        self.outputdir = outputdir
        #options for saving and display of images
        self.show_image = show_image
        self.save_image = save_image

    def createVideo(self, fps=30, interval=1):
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

        # Get the list of image files in the input folder
        image_files = sorted([f for f in os.listdir(self.outputdir) if f.endswith('.png')], key=natural_sort_key)

        # Get the first image to get the width and height information
        first_image = cv2.imread(os.path.join(self.outputdir, image_files[0]))
        height, width, _ = first_image.shape

        # Create a video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        video_writer = cv2.VideoWriter(str(self.outputdir/Path('Animation'+str(fps)+'fps.mp4')), fourcc, fps, (width, height))

        # Write selected images to the video
        for i, image_file in enumerate(image_files):
            if i % interval == 0:
                image_path = os.path.join(self.outputdir, image_file)
                frame = cv2.imread(image_path)
                video_writer.write(frame)

        # Release the video writer object
        video_writer.release()


    def visualiseConfiguration(self,configuration, title='Configuration', textstr=None, snip_dx=0, snip_dy=0):
        xdim = configuration.par['xdim']
        ydim = configuration.par['ydim']
        pic = np.zeros((xdim,ydim,3)) #three RGB channels (RGB values are normalized to 1!)

        #draw volume and surface of cells:
        for cell in configuration.cell_list:         
            if cell.type == 2: #draw the wall
                for pixel in cell.volume_pixel:
                    x = pixel[0]
                    y = pixel[1]
                    color = np.array([130.0, 130.0, 255.0]) #light blue
                    color /= 255.0 #normalize for RGB values between 0 and 1
                    pic[x, y, :] = color
            if cell.type == 1:
                for pixel in cell.volume_pixel:
                    x = pixel[0]
                    y = pixel[1]
                    color = np.array([255.0, 255.0, 255.0]) #white
                    color /= 255.0 #normalize for RGB values between 0 and 1
                    pic[x, y, :] = color

                for pixel in cell.surface_pixel:
                    x = pixel[0]
                    y = pixel[1]
                    color = np.array([130.0, 130.0, 130.0]) #gray
                    color /= 255.0 #normalize for RGB values between 0 and 1
                    pic[x, y, :] = color
                
        pic = cv2.transpose(pic)
        fig, ax = plt.subplots(1)
        print(pic.shape)
        # Optional cuting of the image
        pic = pic[snip_dy:pic.shape[0]-snip_dy, snip_dx:pic.shape[1]-snip_dx, :] 

        fig, ax = plt.subplots()  # Create figure and axes
        ax.imshow(pic, origin="lower")
        plt.axis('off')  # Turn off axis

        if textstr is not None:
            # Define properties for the text box
            props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=1)
            ax.text(0.97, 0.85, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', horizontalalignment='right', bbox=props)


        if self.show_image:
            plt.show()
        if self.save_image:
            filename =  'SimID'+str(configuration.par['SimID'])+'Image'+str(configuration.mcs)+'.pdf'
            plt.savefig(self.outputdir/filename, dpi=300)
            print('Save:', self.outputdir/filename)
        plt.close()

    def visualiseConfigurationCellAge(self,configuration, title='Configuration', textstr='MCS:'):

        def red_color_gradient(age, max_age):
            start_color = np.array([92.0, 12.0, 26.0])      #dark red
            end_color = np.array([250.0, 235.0, 240.0])     #light red
            red = age/max_age * start_color[0] + (1 - age/max_age) * end_color[0]
            green = age/max_age * start_color[1] + (1 - age/max_age) * end_color[1]
            blue = age/max_age * start_color[2] + (1 - age/max_age) * end_color[2]
            color = np.array([red, green, blue])
            return color
        
        xdim = configuration.par['xdim']
        ydim = configuration.par['ydim']
        pic = np.zeros((xdim,ydim,3)) #three RGB channels (RGB values are normalized to 1!)

        #Get Maximum Cell Age:
        max_age = 0
        for cell in configuration.cell_list: 
            if cell.age > max_age: max_age = cell.age

        #draw volume and surface of cells:
        for cell in configuration.cell_list:           
            if cell.type == 2: #draw the wall
                for pixel in cell.volume_pixel:
                    x = pixel[0]
                    y = pixel[1]
                    color = np.array([130.0, 130.0, 255.0]) #light blue
                    color /= 255.0 #normalize for RGB values between 0 and 1
                    pic[x, y, :] = color
            if cell.type == 1:
                for pixel in cell.volume_pixel:
                    x = pixel[0]
                    y = pixel[1]
                    color = red_color_gradient(cell.age, max_age)
                    color /= 255.0 #normalize for RGB values between 0 and 1
                    pic[x, y, :] = color

                for pixel in cell.surface_pixel:
                    x = pixel[0]
                    y = pixel[1]
                    color = np.array([130.0, 130.0, 130.0]) #gray
                    color /= 255.0 #normalize for RGB values between 0 and 1
                    pic[x, y, :] = color
        
        pic = cv2.transpose(pic)
        fig, ax = plt.subplots(1, figsize=(0.9, 0.9))  # Increase figure size to avoid cutting off labels
        ax.imshow(pic, origin="lower")
        #ax = plt.title(title)
        #ax = plt.xlabel('x')
        #ax = plt.ylabel('y')
        plt.axis('off')

        # Add a textbox to the upper right corner of the image
        '''
        props = dict(boxstyle='round', facecolor='white')  # Box style and transparency
        ax.text(
            0.95, 0.90, textstr, transform=ax.transAxes, fontsize=6,
            verticalalignment='top', horizontalalignment='right', bbox=props
        )   
        '''    
        
        if self.show_image:
            plt.show()
        if self.save_image:
            #filename =  'RelativeAgeImage'+str(configuration.mcs)+'.png'
            filename =  'SimID'+str(configuration.par['SimID'])+'Image'+str(configuration.mcs)+'.pdf'
            plt.savefig(self.outputdir/filename, dpi=2000, transparent=True, bbox_inches='tight')
            print('Save:', self.outputdir/filename)
        plt.close()


    

    def visualiseConfigurationPerimeterStrain(self, configuration, curvatures, contours, max_dP=60, curvature_min=-0.02, textstr='MCS:'):
        """
        Visualize the configuration with perimeter strain and curvature using color gradients and custom colorbars.

        Args:
            configuration: The cellular configuration to visualize.
            curvatures: List of curvature values for contours.
            contours: List of contour points for each cell.
            max_dP: Maximum dP value for the strain color gradient.
            curvature_min: Minimum curvature value for color scaling.
            title: Title of the plot.
        """
        def color_gradient(dP, max_dP):
            """
            Create a custom green color gradient for dP values.
            Args:
                dP: Difference between the current and target perimeter.
                max_dP: Maximum perimeter difference value for color scaling.
            Returns:
                A color in RGB format.
            """
            # Define gradient from light green to dark green
            start_color = np.array([230.0, 255.0, 230.0])  # Light green
            end_color = np.array([0.0, 100.0, 0.0])  # Dark green
            if dP >= max_dP:
                return end_color
            if dP <= 0:
                return start_color
            red = dP / max_dP * end_color[0] + (1 - dP / max_dP) * start_color[0]
            green = dP / max_dP * end_color[1] + (1 - dP / max_dP) * start_color[1]
            blue = dP / max_dP * end_color[2] + (1 - dP / max_dP) * start_color[2]
            return np.array([red, green, blue])

        # Create a custom colormap from light green to dark green
        colors = [
            (0.0, np.array([230.0, 255.0, 230.0]) / 255.0),  # Light green
            (1.0, np.array([0.0, 100.0, 0.0]) / 255.0)  # Dark green
        ]
        cmap = LinearSegmentedColormap.from_list("green_gradient", colors)

        # Get the dimensions of the configuration
        xdim = configuration.par['xdim']
        ydim = configuration.par['ydim']
        pic = np.zeros((xdim, ydim, 3))  # Initialize RGB image

        # Draw cells on the image
        for cell in configuration.cell_list:
            if cell.type == 2:  # Wall cells
                for pixel in cell.volume_pixel:
                    x, y = pixel
                    color = np.array([130.0, 130.0, 255.0])  # Light blue
                    pic[x, y, :] = color / 255.0  # Normalize
            elif cell.type == 1:  # Regular cells
                dP = cell.surface - cell.target_surface
                color = color_gradient(dP, max_dP)  # Get color based on dP
                for pixel in cell.volume_pixel:
                    x, y = pixel
                    pic[x, y, :] = color / 255.0  # Normalize
                for pixel in cell.surface_pixel:
                    x, y = pixel
                    color = np.array([130.0, 130.0, 130.0]) / 255.0  # Gray for surface pixels
                    pic[x, y, :] = color

        # Transpose the image for correct orientation
        pic = cv2.transpose(pic)
        # Create the plot
        fig, ax = plt.subplots(1, figsize=(1.2, 1.2))  # Increase figure size to avoid cutting off labels
        ax.imshow(pic, origin="lower")
        plt.axis('off')

        # Add colorbar for dP values using custom colormap
        norm = Normalize(vmin=0, vmax=max_dP)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Needed to avoid warning

        # Create a custom colorbar for dP
        divider = make_axes_locatable(ax)
        width = axes_size.AxesY(ax, aspect=1.0 / 20)  # Adjust aspect ratio
        pad = axes_size.Fraction(1, width)  # Adjust padding
        cax1 = divider.append_axes("left", size=width, pad=pad)
        cbar_dP = plt.colorbar(sm, cax=cax1, label='dP/px')
        cbar_dP.ax.set_ylabel('dP/px', fontsize=8)  # Set label font size
        cbar_dP.ax.tick_params(labelsize=6)  # Adjust tick label font size
        # Move the colorbar label and ticks to the right side
        cbar_dP.ax.yaxis.set_label_position('left')
        cbar_dP.ax.yaxis.tick_left()

        # Visualize curvature values
        vmin = curvature_min
        vmax = -vmin

        # Check if there are contours
        if contours:
            for i, contour in enumerate(contours):
                curvature = curvatures[i]
                x, y = zip(*contour)
                scatter = ax.scatter(x, y, c=curvature, cmap='seismic', vmin=vmin, vmax=vmax, s=0.02)
        else:
            # Create an empty scatter plot with a single point outside the plot area
            scatter = ax.scatter([], [], c=[], cmap='seismic', vmin=vmin, vmax=vmax, s=0.02)

        # Add colorbar for curvature values
        width = axes_size.AxesY(ax, aspect=1.0 / 20)  # Adjust aspect ratio
        pad = axes_size.Fraction(1, width)  # Adjust padding
        cax2 = divider.append_axes("right", size=width, pad=pad)
        cbar_curvature = plt.colorbar(scatter, cax=cax2, label=r'Curvature/px$^{-1}$')
        cbar_curvature.ax.set_ylabel(r'Curvature/px$^{-1}$', fontsize=8)  # Set label font size
        cbar_curvature.ax.tick_params(labelsize=6)  # Adjust tick label font size

        # Adjust colorbar label position to prevent it from getting cut off
        cbar_curvature.ax.yaxis.set_label_position('right')
        #cbar_curvature.ax.yaxis.set_label_coords(4.5, 0.5)  # Shift label inside the plot
        # Reduce the number of ticks on the colorbar
        cbar_curvature.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))


        # Add a textbox to the upper right corner of the image
        props = dict(boxstyle='round', facecolor='white')  # Box style and transparency
        ax.text(
            0.95, 0.95, textstr, transform=ax.transAxes, fontsize=5,
            verticalalignment='top', horizontalalignment='right', bbox=props
)

        # Show or save the image
        if self.show_image:
            plt.show()
        if self.save_image:
            filename = 'PerimeterStrainImage' + str(configuration.mcs) + '.pdf'
            plt.savefig(self.outputdir / filename, dpi=1200, bbox_inches='tight')
            print('Saved:', self.outputdir / filename)
        plt.close()




        

    def visualisePhaseSpaceMacroVol(self, phase_space, mcs, values_x, values_y):
        filepath = self.outputdir / Path("PhaseSpaceMacroVol" + str(mcs) + '.pdf')
        print(filepath)

        fig, ax = plt.subplots(figsize=(1.2, 1.2))
        im = ax.imshow(phase_space, origin="lower", cmap='turbo')
        ax.set_aspect('equal')


        #plt.title('Macroscopic Tissue Volume at $t_{max}$', y=1.06, fontsize=15)
        plt.title('Tissue at MCS: '+str(mcs), y=1.06, fontsize=10)
        plt.xlabel('$J_{CS}$', fontsize=8)
        plt.ylabel('$J_{CM}$', fontsize=8)

        #Labels
        n = 1 #only show every n-th tick
        #'''
        xticks = np.arange(0, len(values_x), n)
        ax.set_xticks(xticks)
        ax.set_xticklabels(values_x[xticks])

        yticks = np.arange(0, len(values_y), n)
        ax.set_yticks(yticks)
        ax.set_yticklabels(values_y[yticks])
        #'''
        ax.tick_params(axis='both', labelsize=5)  # Change both x and y axis tick label size

        # Minor ticks (for position of gridlines)
        ax.set_xticks(np.arange(0.5, len(values_x)-0.5, 1), minor=True)
        ax.set_yticks(np.arange(0.5, len(values_y)-0.5, 1), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        # Remove minor ticks
        ax.tick_params(which='minor', bottom=False, left=False)
        
        '''
        cbar = plt.colorbar()
        plt.clim(0,175000)
        cbar.ax.set_title('$A_{proj}$/px', y=1.03, x=1.08, fontsize=8)
        plt.set_cmap('turbo') #turbo
        '''

        # Use make_axes_locatable for a well-aligned colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Create colorbar
        cbar = fig.colorbar(im, cax=cax)
        #cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.ax.tick_params(labelsize=5)

        # Set color limits (must go before or after imshow, not globally with plt.clim)
        #im.set_clim(-1, 1)

        #cbar.ax.set_ylabel(r'$A_{proj}$/px', fontsize=8, rotation=90)
        # Format colorbar ticks in units of 10^3 px
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*1e-3:.0f}'))

        # Set the colorbar label with proper units
        cbar.ax.set_ylabel(r'$A_{proj} / (\mathrm{px} \cdot 10^3)$', fontsize=8, rotation=90)

        # Position the label on the right side
        cbar.ax.yaxis.set_label_position('right')

        if self.show_image:
            plt.show()
        if self.save_image:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()




    def visualisePhaseSpace(self, phase_space, mcs, values_x, values_y, par_name='Area', target_value=400):

        phase_space = (phase_space-target_value)/target_value
        filepath = self.outputdir / Path("PhaseSpace"+ str(par_name) + str(mcs)+'.pdf')
        print(filepath)

        fig, ax = plt.subplots(figsize=(1.6, 1.6))
        im = ax.imshow(phase_space, origin="lower", cmap='seismic')
        ax.set_aspect('equal')


        #plt.title('Macroscopic Tissue Volume at $t_{max}$', y=1.06, fontsize=15)
        #plt.title(str(par_name) + ' at $t$='+str(mcs), y=1.06, fontsize=15)
        plt.title(str(par_name) + ' Deviation', fontsize=10)
        plt.xlabel('$J_{CS}$', fontsize=8)
        plt.ylabel('$J_{CM}$', fontsize=8)

        #Labels
        n = 5 #only show every n-th tick
        #'''
        xticks = np.arange(0, len(values_x), n)
        ax.set_xticks(xticks)
        ax.set_xticklabels(values_x[xticks])

        yticks = np.arange(0, len(values_y), n)
        ax.set_yticks(yticks)
        ax.set_yticklabels(values_y[yticks])
        ax.tick_params(axis='both', labelsize=6)  # Change both x and y axis tick label size
        #'''

        '''
        xticks = np.arange(0, len(values_x), n)
        ax.set_xticks(xticks)
        # Format y-tick labels to one decimal place
        xtick_labels = [f'{values_x[i]:.1f}' for i in xticks]
        ax.set_xticklabels(xtick_labels)

        yticks = np.arange(0, len(values_y), n)
        ax.set_yticks(yticks)
        # Format y-tick labels to one decimal place
        ytick_labels = [f'{values_y[i]:.1f}' for i in yticks]
        ax.set_yticklabels(ytick_labels)
        '''

        # Minor ticks (for position of gridlines)
        ax.set_xticks(np.arange(0.5, len(values_x)-0.5, 1), minor=True)
        ax.set_yticks(np.arange(0.5, len(values_y)-0.5, 1), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        # Remove minor ticks
        ax.tick_params(which='minor', bottom=False, left=False)
        

        # Use make_axes_locatable for a well-aligned colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Create colorbar
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.ax.tick_params(labelsize=6)

        # Set color limits (must go before or after imshow, not globally with plt.clim)
        im.set_clim(-1, 1)

        if par_name == 'Area':
            cbar.ax.set_ylabel(r'$\Delta A / A_0$', fontsize=8, rotation=90)
        elif par_name == 'Perimeter':
            cbar.ax.set_ylabel(r'$\Delta P / P_0$', fontsize=8, rotation=90)

        # Position the label on the right side
        cbar.ax.yaxis.set_label_position('right')


        if self.show_image:
            plt.show()
        if self.save_image:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()












    
    def visualiseGrowthCurves(self, SimID_list, SimulationsDir, scenario_name):
        Vmax = 1
        normVolData =[]
        for SimID in SimID_list:
            VolumeDataPath = SimulationsDir / Path(str(SimID) + "_Simulation") / Path(scenario_name) / Path("Output") / Path('MacroscopicVolume.txt')
            with open(VolumeDataPath) as data:
                    MacroVol_data = np.genfromtxt(data, dtype='int', delimiter='\t')
            Vol = MacroVol_data[:,1]/Vmax
            normVolData.append(Vol)
        time = MacroVol_data[:,0]


        # plot
        fig, ax = plt.subplots(figsize=(7, 5))
        x = [-240,-245,-250,-255,-260,-265,-270]
        for i, Vol in enumerate(normVolData):
        #for Vol in normVolData:
            ax.plot(time, Vol, label=f"$J_{{CS}}={x[i]}$; $J_{{CM}}=250$") #get label from sim parameters? enumerate?

        #ax.plot(time, Vol1,  'grey', label="$J_{CS}=-70$; $J_{CC}=0$")#, '#130689')
        #ax.plot(time, Vol2, '#1f77b4', label="$J_{CS}=-90$, $J_{CC}=-12.6$")#, '#5601A3') 
        #ax.plot(time, Vol3, 'orange', label="$J_{CS}=-110$, $J_{CC}=-12.6$")#, '#5601A3') 
        #ax.plot(time, Vol4, 'red', label="$J_{CS}=-150$, $J_{CC}=-11.2$")#, '#8104A7') 
        #ax.plot(time, Vol5, 'green', label="$J_{CS}=-150$, $J_{CC}=-12.6$")#, '#D95969')

        ax.set(xlim=(0, 80000), ylim=(0,100000))

        plt.legend(loc="upper left")

        plt.title('Quadratic Pore Growth Curves', y=1.02, fontsize=14)
        plt.xlabel('MCS', fontsize=12)
        plt.ylabel('$A_{proj}$', fontsize=12)

        num_ticks = 5  # Set the number of ticks you want
        y_min, y_max = ax.get_ylim()  # Get current y-axis limits
        ax.set_yticks(np.linspace(y_min, y_max, num_ticks))  # Set evenly spaced ticks
        x_min, x_max = ax.get_xlim()  # Get current y-axis limits
        ax.set_xticks(np.linspace(x_min, x_max, num_ticks))  # Set evenly spaced ticks

        #plt.show()
        plt.tight_layout()
        plt.savefig(self.outputdir / Path('GrowthCurves'), dpi=300)
        plt.close()
    

    def visualiseMeanGrowthCurves(self, List_of_SimID_lists, labels, SimulationsDir, scenario_name, title, parameter_label, colors=['#1f77b4', 'green', 'orange', 'red', 'purple', 'brown', 'pink']):
        N_conditions = len(List_of_SimID_lists)  # Number of conditions (different sets of simulations)
        Vmax = 1  # You can adjust Vmax if needed
        MeanData = []
        StdDevData = []

        for SimID_list in List_of_SimID_lists:
            N_repeats = len(SimID_list)  # Number of repeats of the same simulation
            Data = []  # Store volume data for all simulations under the same condition

            for SimID in SimID_list:
                print('Read SimID: ', SimID)
                VolumeDataPath = SimulationsDir / Path(str(SimID) + "_Simulation") / Path(scenario_name) / Path("Output") / Path('MacroscopicVolume.txt')
                with open(VolumeDataPath) as data:
                    MacroVol_data = np.genfromtxt(data, dtype='int', delimiter='\t')
                Vol = MacroVol_data[:, 1] / Vmax  # Normalize volume data
                print(Vol.shape)
                Data.append(Vol)

            # Convert the list of volume data into a NumPy array (2D: rows are individual simulations)
            Data = np.array(Data)

            # Compute mean and standard deviation along the vertical axis (axis=0)
            mean_array = np.mean(Data, axis=0)
            std_dev_array = np.std(Data, axis=0)

            # Append mean and std deviation data for the current condition
            MeanData.append(mean_array)
            StdDevData.append(std_dev_array)

            # Time data will be the same for all simulations
            time = MacroVol_data[:, 0]

        # Plot mean curves with error bands (std dev)
        fig, ax = plt.subplots(figsize=(1.9, 1.9))#plt.subplots(figsize=(2, 2))

        for i, (mean_vol, std_dev_vol) in enumerate(zip(MeanData, StdDevData)):
            color = colors[i % len(colors)]  # Cycle through colors
            ax.plot(time, mean_vol, label=labels[i], linewidth=0.4, color=color)
            ax.fill_between(time, mean_vol - std_dev_vol, mean_vol + std_dev_vol, alpha=0.1, color=color)

        ax.set(xlim=(-1000, 100001), ylim=(-1000, 125000))  # Limits for Monolayer Figure

        # Assign the legend to a variable
        legend = ax.legend(loc="upper left", ncol=1, columnspacing=0.3, labelspacing=0.3, bbox_to_anchor=(0, 1), fontsize=4, borderpad=0.2, handlelength=0.5)
        legend.get_frame().set_linewidth(0.5)  # Thinner border line
        #legend = ax.legend(loc="upper left")

        plt.title(title, y=1.02, fontsize=10)
        plt.xlabel('t/MCS', fontsize=8)
        #plt.ylabel('$A_{proj}/px$', fontsize=8)

        #'''
        # Update the y-label to reflect new units
        plt.ylabel(r'$A_{proj} / (\mathrm{px} \cdot 10^3)$', fontsize=8)
        # Format y-axis ticks to display in 10^3
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*1e-3:.0f}'))
        #'''

        ax.tick_params(axis='both', labelsize=6)  # Change both x and y axis tick label size
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  

        # Get legend position in figure coordinates
        legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())

        # Convert the legend bbox to axes coordinates
        legend_coords = legend_bbox.transformed(ax.transAxes.inverted())

        # Add a text box to the right of the legend
        if parameter_label is not None:
            #'''
            ax.text(
                legend_coords.x1+0.43, legend_coords.y1-0.015,  # Position to the right of the legend legend_coords.x1 - 0.49, legend_coords.y1-0.33
                parameter_label,
                transform=ax.transAxes,
                fontsize=4,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.2, edgecolor='black', linewidth=0.5)  # Add a white background
            )
            #'''
            '''
            # Add a textbox to the upper right corner of the image
            props = dict(boxstyle='round', facecolor='white')  # Box style and transparency
            ax.text(
                0.95, 0.95, parameter_label, transform=ax.transAxes, fontsize=6,
                verticalalignment='top', horizontalalignment='right', bbox=props)
            '''

        plt.tight_layout()
        #plt.savefig(self.outputdir / Path('MeanGrowthCurves'), dpi=300)
        plt.savefig(self.outputdir / Path('MeanGrowthCurves.pdf'), dpi=300, bbox_inches='tight')
        plt.close()


    def visualiseDataCurves(self, SimID_list, SimulationsDir, scenario_name):
        Data = []
        for SimID in SimID_list:
            CellDataPath = SimulationsDir / Path(str(SimID) + "_Simulation") / Path(scenario_name) / Path("Output") / Path('cell_2.txt')
            with open(CellDataPath) as data:
                    cell_data = np.genfromtxt(data, dtype='int', delimiter='\t')
            D = cell_data[:,2]
            Data.append(D)
        time = cell_data[:,0]

        # plot
        fig, ax = plt.subplots(figsize=(5, 5))
        #for P in PerimeterData:
            #ax.plot(time, Vol, label="???") #get label from sim parameters? enumerate?


        ax.plot(time, Data[3],  '#1f77b4', label="$\sigma = 1.0$", linewidth=0.9)
        ax.plot(time, Data[2],  'green', label="$\sigma = 0.9$", linewidth=0.9)
        ax.plot(time, Data[1],  'purple', label="$\sigma = 0.8$", linewidth=0.9)
        ax.plot(time, Data[0],  'violet', label="$\sigma = 0.7$", linewidth=0.9)

        '''
        ax.plot(time, PerimeterData[0],  'grey', label="$J_{CS}=0$", linewidth=0.9)#, '#130689')
        ax.plot(time, PerimeterData[1],  '#1f77b4', label="$J_{CS}=-50$", linewidth=0.9)#, '#130689')
        ax.plot(time, PerimeterData[2],  'orange', label="$J_{CS}=-100$", linewidth=0.9)#, '#130689')
        ax.plot(time, PerimeterData[3],  'red', label="$J_{CS}=-150$", linewidth=0.9)#, '#130689')
        '''



        #ax.plot(time, Vol2, '#1f77b4', label="$J_{CS}=-90$, $J_{CC}=-12.6$")#, '#5601A3') 
        #ax.plot(time, Vol3, 'orange', label="$J_{CS}=-110$, $J_{CC}=-12.6$")#, '#5601A3') 
        #ax.plot(time, Vol4, 'red', label="$J_{CS}=-150$, $J_{CC}=-11.2$")#, '#8104A7') 
        #ax.plot(time, Vol5, 'green', label="$J_{CS}=-150$, $J_{CC}=-12.6$")#, '#D95969')

        ax.set(xlim=(0, 500), ylim=(395, 405))

        plt.legend(loc="upper left")

        ax.text(0.95, 0.95, "$J_{CS}=-50$", transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

        plt.title('Cell Area', y=1.02, fontsize=14)
        plt.xlabel('t/mcs', fontsize=12)
        plt.ylabel('$P/px$', fontsize=12)

        #plt.show()
        plt.tight_layout()
        plt.savefig(self.outputdir / Path('AreaCurves'), dpi=300)
        plt.close()

    
    def visualiseCellStatisticsCurves(self, List_of_SimID_lists, SimulationsDir, scenario_name, par_name='Perimeter', ymin=0, ymax=500, target_value = 90.25334930419922):
        N_conditions = len(List_of_SimID_lists)
        MeanData = []
        StdDevData = []
        for SimID_list in List_of_SimID_lists:
            N_repeats = len(SimID_list) # Number of repeats of the same simulation
            Data = []
            for SimID in SimID_list:
                print('Read SimID: ', SimID)
                CellDataPath = SimulationsDir / Path(str(SimID) + "_Simulation") / Path(scenario_name) / Path("Output") / Path('cell_2.txt')
                with open(CellDataPath) as data:
                        cell_data = np.genfromtxt(data, dtype='int', delimiter='\t')
                if par_name=='Area':
                    D = cell_data[:,2]
                if par_name=='Perimeter':
                    D = cell_data[:,3]
                Data.append(D)

            # Convert PerimeterData to a NumPy array (2D array where rows are individual arrays)
            Data = np.array(Data)

            Data = (Data-target_value)/target_value

            # Compute the mean and standard deviation along the vertical axis (axis=0 means column-wise)
            mean_array = np.mean(Data, axis=0)
            MeanData.append(mean_array)
            std_dev_array = np.std(Data, axis=0)
            StdDevData.append(std_dev_array)

            time = cell_data[:,0]

        # plot
        fig, ax = plt.subplots(figsize=(2, 2))
        #for P in PerimeterData:
            #ax.plot(time, Vol, label="???") #get label from sim parameters? enumerate?
        '''
        ax.plot(time, MeanPerimeter[3],  'red', label="$J_{CS}=-150$", linewidth=0.9)
        ax.fill_between(time, MeanPerimeter[3] - StdDevPerimeter[3], MeanPerimeter[3] + StdDevPerimeter[3], color='red', alpha=0.1)
        ax.plot(time, MeanPerimeter[2],  'orange', label="$J_{CS}=-100$", linewidth=0.9)
        ax.fill_between(time, MeanPerimeter[2] - StdDevPerimeter[2], MeanPerimeter[2] + StdDevPerimeter[2], color='orange', alpha=0.1)
        ax.plot(time, MeanPerimeter[1],  '#1f77b4', label="$J_{CS}=-50$", linewidth=0.9)
        ax.fill_between(time, MeanPerimeter[1] - StdDevPerimeter[1], MeanPerimeter[1] + StdDevPerimeter[1], color='#1f77b4', alpha=0.1)
        ax.plot(time, MeanPerimeter[0],  'grey', label="$J_{CS}=0$", linewidth=0.9)
        ax.fill_between(time, MeanPerimeter[0] - StdDevPerimeter[0], MeanPerimeter[0] + StdDevPerimeter[0], color='grey', alpha=0.1)
        '''

        colors = ['red','orange','#1f77b4','grey']

        ax.plot(time, MeanData[3],  colors[0], label="$J_{CS}{=}{-50}$, $J_{CM}{=}{50}$", linewidth=0.4)
        ax.fill_between(time, MeanData[3] - StdDevData[3], MeanData[3] + StdDevData[3], color=colors[0], alpha=0.1)
        ax.plot(time, MeanData[2],  colors[1], label="$J_{CS}{=}{-50}$, $J_{CM}{=}{0}$", linewidth=0.4)
        ax.fill_between(time, MeanData[2] - StdDevData[2], MeanData[2] + StdDevData[2], color=colors[1], alpha=0.1)
        ax.plot(time, MeanData[1],  colors[2], label="$J_{CS}{=}{0}$, $J_{CM}{=}{50}$", linewidth=0.4)
        ax.fill_between(time, MeanData[1] - StdDevData[1], MeanData[1] + StdDevData[1], color=colors[2], alpha=0.1)
        ax.plot(time, MeanData[0],  colors[3], label="$J_{CS}{=}{0}$, $J_{CM}{=}{0}$", linewidth=0.4)
        ax.fill_between(time, MeanData[0] - StdDevData[0], MeanData[0] + StdDevData[0], color=colors[3], alpha=0.1)

        plt.legend(loc="upper right", bbox_to_anchor=(1, 0.93), fontsize=5, handlelength=0.5)

        # Add a text box in the upper right corner
        #ax.text(0.95, 0.95, "$J_{CS}=-50$", transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

        ax.tick_params(axis='both', labelsize=6)  # Change both x and y axis tick label size
        plt.title('Average '+ par_name + ' Deviation', y=1.02, fontsize=10)
        plt.xlabel('t/MCS', fontsize=8)


        if par_name == 'Area':
            plt.ylabel(r'$\Delta A$/$A_0$', fontsize=8)
        if par_name == 'Perimeter':
            plt.ylabel(r'$\Delta P$/$P_0$', fontsize=8)
        

        #plt.show()
        plt.tight_layout()
        plt.savefig(self.outputdir / Path(par_name+'StatisticCurves.pdf'), dpi=300, bbox_inches='tight')
        plt.close()