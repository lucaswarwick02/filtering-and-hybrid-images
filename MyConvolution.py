import numpy as np

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders
    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray
    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    
    # Flip the template around both axes
    inverted_template = kernel[::-1,::-1]
    
    x, y = len(image), len(image[0])
    offset = int((len(inverted_template) - 1) / 2)
    
    
    def convolve_2d (image_2d, inverted_template):
        padded_image_2d = np.zeros((x + len(image_2d) - 1, y + len(image_2d) - 1)) # Create a zeroed array for padding the image
        padded_image_2d[offset:offset+x, offset:offset+y] = image_2d # Add the image into the center of the padded image
        
        new_image_2d = np.zeros((x, y)).astype(int)
        for row in range(0+offset, x+offset):
            for column in range(0+offset, y+offset):
                section = np.array(padded_image_2d[row-offset:row+offset+1,column-offset:column+offset+1])
                new_image_2d[row-offset][column-offset] = np.sum(np.multiply(section, inverted_template))
                
        return new_image_2d
    
    
    if image.ndim == 2:
        # Pad the pixel values to become arrays, can then iterate through
        return convolve_2d(image, inverted_template)

    # Now image can be looped through
    new_image = np.zeros(shape=np.shape(image)).astype(int)

        
    # For each color index...
    for color_index in range(0, image.shape[2]):
        # ... Grab the single 2D array
        color_array = image[:,:,color_index]
        
        # padded_color_array = np.zeros((x + len(inverted_template) - 1, y + len(inverted_template) - 1)) # Create a zeroed array for padding the image
        # padded_color_array[offset:offset+x, offset:offset+y] = color_array # Add the image into the center of the padded image
        
        # new_color_array = np.zeros((x, y)).astype(int)
        # for row in range(0+offset, x+offset):
        #     for column in range(0+offset, y+offset):
        #         section = np.array(padded_color_array[row-offset:row+offset+1,column-offset:column+offset+1])
        #         new_color_array[row-offset][column-offset] = np.sum(np.multiply(section, inverted_template))
        
                
        # new_image[:,:,color_index] = new_color_array # Syntax grabs the first element from the subsets, learnt the syntax in Machine Learning Technologies!
        new_image[:,:,color_index] = convolve_2d(color_array, inverted_template)

    return new_image
