import math
import numpy as np

from MyConvolution import convolve

def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.
    
    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    
    :param lowSigma: the standard deviation of the Gaussian used for low-pass filteringlowImage
    :type float
    
    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    
    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float
    
    :returns returns the hybrid image created by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining
        it with a high-pass image created by subtracting highImage from highImage convolved with a Gaussian of s.d. highSigma.
        The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """
    if lowImage.ndim == 2:
        # Pad the pixel values to become arrays, can then iterate through
        lowImage = lowImage.reshape((lowImage.shape[0], lowImage.shape[1], 1))
    if highImage.ndim == 2:
        # Pad the pixel values to become arrays, can then iterate through
        highImage = highImage.reshape((highImage.shape[0], highImage.shape[1], 1))
        
    low_pass_filter = makeGaussianKernel(sigma=lowSigma)
    low_pass_image = convolve(lowImage, low_pass_filter)

    high_pass_filter = makeGaussianKernel(sigma=highSigma)
    high_pass_image = highImage - convolve(highImage, high_pass_filter)

    hybrid_image = low_pass_image + high_pass_image

    return np.clip(hybrid_image, 0, 255)
    
 
def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    size = int(math.floor(8*sigma+1))
    if (size % 2 == 0):
        # Size is even, make odd
        size += 1

    # If size == 1, return unit kernel
    if size == 1:
        return np.array([[1]])

    kernel = np.zeros((size, size))

    cum_sum = 0
    x_range = int((size - 1) / 2)
    y_range = int((size - 1) / 2)

    for x in range(-x_range, x_range + 1):
        for y in range(-y_range, y_range + 1):
            kernel[y + y_range][x + x_range] = (math.exp(-((x * x) + (y * y)) / (2 * sigma * sigma))) / (2 * math.pi * sigma * sigma)
            cum_sum += kernel[y + y_range][x + x_range]
            
    kernel /= cum_sum

    return kernel