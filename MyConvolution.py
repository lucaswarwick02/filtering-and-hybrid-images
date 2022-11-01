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
        image = np.array(image)
    else:
        image = image.transpose((2, 0, 1))

    # Now image can be looped through
    new_image = np.zeros(shape=np.shape(image)).astype(int)

    for index, color_array in enumerate(image):
        padded_color_array = np.pad(color_array, [offset, offset], mode='constant')
        new_color_array = np.zeros((x, y)).astype(int)
        for row in range(0+offset, x+offset):
            for column in range(0+offset, y+offset):
                section = np.array(padded_color_array[row-offset:row+offset+1,column-offset:column+offset+1])
                new_color_array[row-offset][column-offset] = np.sum(np.multiply(section, inverted_template))

        new_image[index] = new_color_array

    return new_image.transpose((1, 2, 0))
