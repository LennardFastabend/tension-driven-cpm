import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

def array_to_pif(np_array, output_file):
    """
    Converts a NumPy array into a PIF file format, writing one line per individual pixel.
    
    - Medium pixels (value 0 in the array) are ignored.
    - Wall pixels (value 1 in the array) are written as `0 Wall ...`.
    - Cell pixels (values 2, 3, 4,...) are written as `<cell_id - 1> Cell ...`.
    
    Arguments:
    np_array : 2D numpy array where:
               - 0 represents medium pixels (ignored)
               - 1 represents wall pixels
               - 2, 3, 4,... represent cell IDs
    output_file : Path to the output PIF file
    """
    # Get the dimensions of the numpy array
    height, width = np_array.shape
    
    with open(output_file, 'w') as pif_file:
        for y in range(height):
            for x in range(width):
                cell_id = np_array[y, x]
                
                if cell_id == 0:
                    # Medium pixels are ignored (no output for medium)
                    continue
                
                elif cell_id == 1:
                    # Wall pixels (value 1), write as `0 Wall ...`
                    pif_file.write(f"0 Wall {x} {x+1} {y} {y+1} 0 0\n")
                
                else:
                    # Cell pixels (values 2, 3, 4,...), write as `cell_id - 1`
                    pif_file.write(f"{cell_id - 1} Cell {x} {x+1} {y} {y+1} 0 0\n")
    
    print(f"PIF file written to {output_file}")

def create_triangle_mask(size, pos1, pos2, pos3):
    """
    Creates a mask for an equilateral triangle defined by three vertices.
    
    Arguments:
    size : int - the size of the square array (size x size).
    pos1 : list - coordinates of the first vertex [x, y].
    pos2 : list - coordinates of the second vertex [x, y].
    pos3 : list - coordinates of the third vertex [x, y].
    
    Returns:
    np.array - an array with the triangle's interior filled with zeros.
    """
    # Create an array filled with ones (walls)
    mask = np.ones((size, size), dtype=int)
    
    # Define the vertices
    vertices = np.array([pos1, pos2, pos3])
    
    # Get the bounding box of the triangle
    min_x = int(np.min(vertices[:, 0]))
    max_x = int(np.max(vertices[:, 0]))
    min_y = int(np.min(vertices[:, 1]))
    max_y = int(np.max(vertices[:, 1]))
    
    # Check each pixel in the bounding box to see if it's inside the triangle
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            # Barycentric coordinates method to check if the point (x, y) is inside the triangle
            v0 = vertices[1] - vertices[0]
            v1 = vertices[2] - vertices[0]
            v2 = np.array([x, y]) - vertices[0]
            
            # Compute dot products
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)
            
            # Barycentric coordinates
            invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom
            
            # Check if the point is in the triangle
            if (u >= 0) and (v >= 0) and (u + v <= 1):
                mask[y, x] = 0  # Fill the interior of the triangle with zeros

    return mask

def create_rectangle_mask(size, pos1, pos2, pos3, pos4):
    """
    Creates a mask for a rectangle defined by four corner vertices.
    
    Arguments:
    size : int - the size of the square array (size x size).
    pos1, pos2, pos3, pos4 : list - coordinates of the four vertices [x, y].
    
    Returns:
    np.array - an array with the rectangle's interior filled with zeros.
    """
    # Create an array filled with ones (walls)
    mask = np.ones((size, size), dtype=int)
    
    # Extract x and y coordinates
    x_coords = [pos1[0], pos2[0], pos3[0], pos4[0]]
    y_coords = [pos1[1], pos2[1], pos3[1], pos4[1]]
    
    # Get the bounding box of the rectangle
    min_x = int(np.min(x_coords))
    max_x = int(np.max(x_coords))
    min_y = int(np.min(y_coords))
    max_y = int(np.max(y_coords))
    
    # Fill the rectangle area with zeros
    mask[min_y:max_y+1, min_x:max_x+1] = 0
    
    return mask


def create_cleft_mask(size, alpha):
    """
    Creates a mask with a cleft (triangular opening) defined by an angle alpha.

    Arguments:
    size : int - the size of the square array (size x size).
    alpha : float - the angle (in degrees) defining the width of the cleft.

    Returns:
    np.array - an array with the cleft's interior filled with zeros.
    """
    # Create an array filled with ones (walls)
    mask = np.ones((size, size), dtype=int)

    # Define the tip of the cleft (triangle's apex)
    pos0 = np.array([20, size // 2])

    # Convert alpha to radians
    alpha_rad = math.radians(alpha)
    
    # Calculate the length of the base using the angle and a fixed height
    height = size-pos0[0]
    
    # Correct calculation for half the base of the triangle based on alpha
    half_base = height * math.tan(alpha_rad / 2)

    # Define the base points of the triangle (left and right)
    pos1 = np.array([size, int(pos0[1] - half_base)])  # Left base point
    pos2 = np.array([size, int(pos0[1] + half_base)])  # Right base point

    # Helper function to check if a point is inside the triangle
    def is_point_in_triangle(p, v1, v2, v3):
        # Using the area method to check if the point is inside the triangle
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(p, v1, v2)
        d2 = sign(p, v2, v3)
        d3 = sign(p, v3, v1)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)

    # Loop through all pixels in the mask
    for i in range(size):
        for j in range(size):
            # Check if the current pixel (i, j) is inside the triangle
            if is_point_in_triangle(np.array([i, j]), pos0, pos1, pos2):
                mask[i, j] = 0  # Set the pixel inside the triangle to zero

    return mask

def create_capillary_mask(size, width):
    """
    Creates a mask with a rectangular capillary (open at the top).

    Arguments:
    size : int - the size of the square array (size x size).
    width : int - the width of the capillary.

    Returns:
    np.array - an array with the capillary's interior filled with zeros.
    """
    # Create an array filled with ones (walls)
    mask = np.ones((size, size), dtype=int)
    
    # Calculate the left and right bounds of the capillary
    left = (size - width) // 2
    right = left + width
    
    # Set the capillary interior to zero (open region), keeping a 20px gap at the bottom
    mask[20:size, left:right] = 0
    
    return mask

def create_rotated_beam_mask(size, thickness, alpha):
    """
    Creates a mask with only a thick horizontal beam through the center, rotated by an angle.
    
    Arguments:
    size : int - size of the square array (size x size).
    thickness : int - thickness of the beam.
    alpha : float - angle in degrees (0° = horizontal, 90° = vertical).
    
    Returns:
    np.array - an array with the beam's interior filled with zeros.
    """
    # Create an array filled with zeros (background)
    mask = np.zeros((size, size), dtype=int)
    
    # Create a blank image for drawing
    beam = np.zeros((size, size), dtype=np.uint8)
    
    # Define beam rectangle
    center = (size // 2, size // 2)
    rect_size = (size, thickness)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, alpha, 1.0)
    
    # Draw a filled rectangle in the center
    cv2.rectangle(beam, (0, size//2 - thickness//2), (size, size//2 + thickness//2), 255, -1)
    
    # Rotate the beam
    rotated_beam = cv2.warpAffine(beam, rotation_matrix, (size, size), flags=cv2.INTER_NEAREST)
    
    # Convert to binary mask (1 = beam, 0 = background)
    mask[rotated_beam == 255] = 1
    
    return mask

'''
# Triangle definition
size = 599 # Size of the array
l = size-20 # Side length of the triangle
h = l * (np.sqrt(3) / 2)  # Height of the equilateral triangle
y_offset=50
pos1 = [10, 1 + y_offset]  # Lower left corner
pos2 = [l+10, 1+ y_offset]  # Lower right corner
pos3 = [10+l/2, 1 + h+ y_offset]  # Upper corner/tip of triangle

# Create the triangle mask
triangle_mask = create_triangle_mask(size, pos1, pos2, pos3)
#print(triangle_mask)
# Visualize the mask
plt.imshow(triangle_mask, cmap='gray')
plt.title("Triangle Mask")
plt.show()

# Convert the array to a PIF file
array_to_pif(triangle_mask, "output.piff")
'''
'''
#Cleft Definition
size = 599
alpha = 67.5 #degree

cleft_mask = create_cleft_mask(size, alpha)
cleft_mask[300:,:] = 0
#cleft_mask = np.fliplr(cleft_mask)


# Convert the array to a PIF file
array_to_pif(cleft_mask, 'cleft67_5'+ '.piff')

plt.imshow(cleft_mask, cmap='gray', origin='lower')
plt.title("Cleft Mask")
plt.show()
'''

'''
#Serrated Edges Definition
size = 599
thickness = 20

for alpha in np.arange(0,100,15):
    mask = create_rotated_beam_mask(size, thickness, alpha)


    # Convert the array to a PIF file
    array_to_pif(mask, 'beam'+ str(alpha) + '.piff')

    plt.imshow(mask, cmap='gray', origin='lower')
    plt.title("Serrated Edge Mask")
    plt.show()
'''
#Capillary Definition
size = 599
width = 250

for width in np.arange(50,300,50):

    mask = create_capillary_mask(size, width)


    # Convert the array to a PIF file
    array_to_pif(mask, 'capillary'+ str(width) + '.piff')

    plt.imshow(mask, cmap='gray', origin='lower')
    plt.title("Capillary Mask")
    plt.show()