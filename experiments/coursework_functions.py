import math
import numpy as np


def print_2d_array(array_2d: [[float]], decimal_places=4):
    print('\n'.join([''.join([('{:.' + str(decimal_places) + 'f} ').format(item) for item in row]) for row in array_2d]))


def create_guassian_filter(n: int, m: int, sigma=1.0):
    s = 2.0 * sigma * sigma

    guassian_filter = [ [0]*n for i in range(m) ]

    sum = 0
    x_range = int((n - 1) / 2)
    y_range = int((m - 1) / 2)

    for x in range(-x_range, x_range + 1):
        for y in range(-y_range, y_range + 1):
            r = math.sqrt((x * x) + (y * y))
            guassian_filter[y + y_range][x + x_range] = (math.exp(-(r *r) / 2)) / (math.pi * s)
            sum += guassian_filter[y + y_range][x + x_range]

    for x in range(-x_range, x_range + 1):
        for y in range(-y_range, y_range + 1):
            guassian_filter[y + y_range][x + x_range] /= sum

    return guassian_filter


def apply_kernel(image: [[float]], kernel: [[float]]):
    x, y = len(image), len(image[0])
    n, m = len(kernel), len(kernel[0])
    new_x, new_y = x - n + 1, y - m + 1

    new_image = [[0]*new_y for i in range(new_x)]

    for row_offset in range(0, new_x):
        for column_offset in range(0, new_y):
            sum = 0
            for row in range(0, n):
                for column in range(0, m):
                    sum += image[row + row_offset][column + column_offset] * kernel[row][column]
            new_image[row_offset][column_offset] = sum

    return new_image
