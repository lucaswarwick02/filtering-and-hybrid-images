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
    inverted_template = np.flip(kernel)
    
    x, y = len(image), len(image[0])
    offset = int((len(kernel) - 1) / 2)
    
    if image.ndim == 2:
        # Pad the pixel values to become arrays, can then iterate through
        image = image.reshape((image.shape[0], image.shape[1], 1))

    # Now image can be looped through
    new_image = np.zeros(shape=np.shape(image)).astype(int)
        
    for color_index in range(0, image.shape[2]):
        color_array = image[:,:,color_index]
        padded_color_array = np.pad(color_array, [offset, offset], mode='constant')
        new_color_array = np.zeros((x, y)).astype(int)
        for row in range(0+offset, x+offset):
            for column in range(0+offset, y+offset):
                section = np.array(padded_color_array[row-offset:row+offset+1,column-offset:column+offset+1])
                new_color_array[row-offset][column-offset] = np.sum(np.multiply(section, inverted_template))
                
        new_image[:,:,color_index] = new_color_array # Syntax grabs the first element from the subsets, learnt the syntax in Machine Learning Technologies!

    return new_image
