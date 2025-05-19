import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def get_macroscopic_volume(mcs, SimID_Outputdir):
    #Read the corresponding MacroscopicVolume Output
    filepath = SimID_Outputdir / Path('MacroscopicVolume.txt')

    with open(filepath) as data:
        MacroVol_data = np.genfromtxt(data, dtype='int', delimiter='\t')
    '''
    MacroVol_data[:,0] is mcs
    MacroVol_data[:,1] is MacroVol
    '''
    # find the corresponding Row of the given MCS
    timepoints = list(MacroVol_data[:,0])
    row_index = timepoints.index(mcs)
    # save and return the MacroscopicTissueVolume
    MacroVol = MacroVol_data[row_index,1]
    return MacroVol

def get_cell_volume_surface(mcs, SimID_Outputdir, cell_id):
    # Read the corresponding MacroscopicVolume Output
    filepath = SimID_Outputdir / Path('cell_' + str(cell_id) + '.txt')
    
    with open(filepath) as data:
        cell_data = np.genfromtxt(data, dtype='float', delimiter='\t')
    
    # Ensure cell_data is 2-dimensional
    if cell_data.ndim == 1:
        cell_data = cell_data[np.newaxis, :]  # Convert to 2D array with one row
    
    '''
    cell_data[:,0] is mcs
    cell_data[:,1] is cell.type
    cell_data[:,2] is volume
    cell_data[:,3] is surface
    cell_data[:,4] is target volume
    cell_data[:,5] is target surface
    '''
    # Find the corresponding row of the given MCS
    timepoints = list(cell_data[:, 0])
    
    # Check if mcs is in timepoints
    if mcs in timepoints:
        row_index = timepoints.index(mcs)
        cell_volume = cell_data[row_index, 2]
        cell_surface = cell_data[row_index, 3]
    else:
        # If mcs is not in timepoints, set volume and surface to zero
        cell_volume = 0
        cell_surface = 0

    return cell_volume, cell_surface


def get_tissue_medium_contour(config):
    tissue_medium_contour = []
    # Loop through all the cells
    for cell in config.cell_list:           
        if cell.type == 1:  # Cells
            for pixel in cell.surface_pixel:
                x = pixel[0]
                y = pixel[1]
                # Check the first-order neighbors (up, down, left, right)
                neighbors = [
                    (x, y - 1), 
                    (x, y + 1),
                    (x - 1, y), 
                    (x + 1, y)]
                


                # Check if any neighbor has a value of 0
                has_zero_neighbor = any(
                    config.pixel_grid[nx][ny] == 0  # Adjust this based on your grid structure
                    for nx, ny in neighbors
                    if 0 <= nx < config.par["xdim"] and 0 <= ny < config.par["ydim"]  # Ensure within bounds
                )

                # If there's at least one zero neighbor, add to the list
                if has_zero_neighbor:
                    tissue_medium_contour.append((x, y))

    return tissue_medium_contour

def sort_pixels_neighborhood_with_starting_point(pixels, neighborhood_radius=np.sqrt(2)):
    """
    Sort a list of pixels based on geometric neighborhood (nearest neighbor),
    starting with a pixel that has only one direct neighbor if possible.
    
    Parameters:
    - pixels: List of (x, y) tuples representing the pixel coordinates.
    - neighborhood_radius: Distance threshold to consider a pixel a direct neighbor.
    
    Returns:
    - sorted_pixels: List of (x, y) tuples ordered by geometric connectivity.
    """
    if not pixels:
        return []
    
    # Convert to numpy array for efficient computation
    pixels = np.array(pixels)
    
    # Step 1: Identify a starting pixel with only one direct neighbor
    starting_pixel = None
    for i, pixel in enumerate(pixels):
        distances = np.linalg.norm(pixels - pixel, axis=1)
        # Count neighbors within the neighborhood radius (excluding self)
        neighbor_count = np.sum((distances <= neighborhood_radius) & (distances > 0))
        if neighbor_count == 1:
            starting_pixel = tuple(pixel)  # Convert to tuple for compatibility with the list
            break
    
    # If no such pixel is found, default to the first pixel
    if starting_pixel is None:
        starting_pixel = tuple(pixels[0])  # Convert to tuple
    
    # Initialize sorted_pixels and remaining list
    sorted_pixels = [starting_pixel]
    remaining = [tuple(p) for p in pixels if tuple(p) != starting_pixel]  # Remove the starting pixel
    
    # Step 2: Sort the remaining pixels based on nearest neighbor
    while remaining:
        # Find the nearest pixel to the last added pixel
        last_pixel = np.array(sorted_pixels[-1])  # Convert back to array for distance computation
        distances = np.linalg.norm(np.array(remaining) - last_pixel, axis=1)
        nearest_idx = np.argmin(distances)
        
        # Add the nearest pixel to the sorted list and remove it from remaining
        sorted_pixels.append(remaining.pop(nearest_idx))
    
    return np.array(sorted_pixels)


def separate_contours(sorted_pixels, distance_threshold=np.sqrt(2)):
    """
    Separate sorted pixels into multiple contours based on a distance threshold.
    
    Parameters:
    - sorted_pixels: Numpy array of (x, y) coordinates sorted by geometric connectivity.
    - distance_threshold: Maximum allowed distance between consecutive points in the same contour.
    
    Returns:
    - contours: List of numpy arrays, each representing a separate contour.
    """
    if len(sorted_pixels) == 0:
        return []
    
    contours = []
    current_contour = [sorted_pixels[0]]  # Start the first contour with the first pixel
    
    for i in range(1, len(sorted_pixels)):
        # Calculate the distance between the current and previous pixel
        distance = np.linalg.norm(sorted_pixels[i] - sorted_pixels[i - 1])
        
        if distance > distance_threshold:
            # If the distance exceeds the threshold, save the current contour and start a new one
            contours.append(np.array(current_contour))
            current_contour = []
        
        current_contour.append(sorted_pixels[i])
    
    # Append the last contour
    if current_contour:
        contours.append(np.array(current_contour))

    # Check Possible connectivity between separated contours
    
    return contours

def calculate_curvature(contour, scale, image):
    """
    Calculate the curvature for each pixel of a sorted contour.
    
    Parameters:
        contour (array): List of contour points as 2D coordinates.
        scale (int): Distance to walk along the contour.
        image (2D array): Optional, image to determine the sign of curvature.
    
    Returns:
        curvatures (list): List of curvatures for each pixel in the contour.
    """

    def calculate_circle(p1, p2, p3):
        """
        Calculate the center and radius of a circle passing through three points.
        """
        # Constructing the determinant method for the circle
        A = np.array([
            [p1[0], p1[1], 1],
            [p2[0], p2[1], 1],
            [p3[0], p3[1], 1],
        ])
        B = np.array([
            [p1[0]**2 + p1[1]**2, p1[1], 1],
            [p2[0]**2 + p2[1]**2, p2[1], 1],
            [p3[0]**2 + p3[1]**2, p3[1], 1],
        ])
        C = np.array([
            [p1[0]**2 + p1[1]**2, p1[0], 1],
            [p2[0]**2 + p2[1]**2, p2[0], 1],
            [p3[0]**2 + p3[1]**2, p3[0], 1],
        ])
        D = np.array([
            [p1[0]**2 + p1[1]**2, p1[0], p1[1]],
            [p2[0]**2 + p2[1]**2, p2[0], p2[1]],
            [p3[0]**2 + p3[1]**2, p3[0], p3[1]],
        ])
        detA = np.linalg.det(A)
        detB = np.linalg.det(B)
        detC = np.linalg.det(C)
        detD = np.linalg.det(D)

        # Circle center and radius
        if detA == 0:
            return None, None  # Collinear points
        x_center = 0.5 * detB / detA
        y_center = -0.5 * detC / detA
        radius = np.sqrt((x_center - p1[0])**2 + (y_center - p1[1])**2)
        return (x_center, y_center), radius

    curvatures = []
    n = len(contour)
    
    # Check if first and last points are direct neighbors
    is_cyclic = np.linalg.norm(contour[0] - contour[-1]) <= np.sqrt(2)
    
    for i in range(n):
        # Skip curvature calculation near start and end if not cyclic
        if not is_cyclic and (i < scale or i >= n - scale):
            curvatures.append(0)
            continue

        # Get neighboring indices, wrapping around if cyclic
        prev_idx = (i - scale) % n if is_cyclic else max(0, i - scale)
        next_idx = (i + scale) % n if is_cyclic else min(n - 1, i + scale)
        p1, p2, p3 = contour[prev_idx], contour[i], contour[next_idx]
        
        # Calculate the circle center and radius
        center, radius = calculate_circle(p1, p2, p3)

        if radius is None or radius >= 1000:  # Filter out very small curvature values
            curvature = 0
        else:
            curvature = -1 / radius  # Concave curvature is defined as negative
        
        # Determine the sign of curvature using the image
        if curvature != 0:
            mid_x = int((p1[0] + p3[0]) // 2)
            mid_y = int((p1[1] + p3[1]) // 2)
            if image[mid_x, mid_y] != 0:  # Example condition for curvature sign
                curvature = -curvature  # Convex curvature is defined as positive
        curvatures.append(curvature)
    
    return np.array(curvatures)