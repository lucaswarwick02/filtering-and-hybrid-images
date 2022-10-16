import math
import numpy as np


def print_2d_array(array_2d: [[float]], decimal_places=4):
    print(
        '\n'.join([''.join([('{:.' + str(decimal_places) + 'f} ').format(item) for item in row]) for row in array_2d]))


def create_guassian_filter(sigma):
    s = 2.0 * sigma * sigma
    n = m = int(8 * sigma + 1)

    guassian_filter = [[0] * n for i in range(m)]

    cum_sum = 0
    x_range = int((n - 1) / 2)
    y_range = int((m - 1) / 2)

    for x in range(-x_range, x_range + 1):
        for y in range(-y_range, y_range + 1):
            r = math.sqrt((x * x) + (y * y))
            guassian_filter[y + y_range][x + x_range] = (math.exp(-(r * r) / 2)) / (math.pi * s)
            cum_sum += guassian_filter[y + y_range][x + x_range]

    for x in range(-x_range, x_range + 1):
        for y in range(-y_range, y_range + 1):
            guassian_filter[y + y_range][x + x_range] /= cum_sum

    return guassian_filter


def invert_template(original_template):
    inverted_template = original_template.copy()
    inverted_template.reverse()

    for row in inverted_template:
        row.reverse()

    return inverted_template


def perform_convolution(image: [[float]], template: [[float]]):
    # Flip the template around both axes
    inverted_template = invert_template(template)

    x, y = len(image), len(image[0])
    n, m = len(inverted_template), len(inverted_template[0])
    new_x, new_y = x - n + 1, y - m + 1

    new_image = [[0] * new_y for i in range(new_x)]

    for row_offset in range(0, new_x):
        for column_offset in range(0, new_y):
            sum = 0
            for row in range(0, n):
                for column in range(0, m):
                    sum += image[row + row_offset][column + column_offset] * inverted_template[row][column]
            new_image[row_offset][column_offset] = sum

    return new_image


def red_text(text: str):
    return "\x1b[31m\"" + text + "\"\x1b[0m"


def green_text(text: str):
    return "\x1b[32m\"" + text + "\"\x1b[0m"


def test_text(test_name: str, bool_exp: bool):
    if bool_exp:
        return green_text(f'{test_name}: PASSED')
    else:
        return red_text(f'{test_name}: FAILED')
