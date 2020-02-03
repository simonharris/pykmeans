"""
Standardise the data with range

Assuming matrix is an np array you can just do
matrix = (matrix - matrix.mean(axis=0))/(matrix.max(axis=0) - matrix.min(axis=0))

The range for each column should then be one. In other words,
matrix.max(axis=0) - matrix.min(axis=0)
should return as many ones as there are columns.

If you think of the data as a table in excel, the above the same as:
[new cell] = ([old cell] - [mean of old cell's column]) /
  ([highest value of old cell's column] - [lowest value of old cell's column])
"""

import numpy as np


def process(matrix):
    """Standardise as per above comment"""

    matrix = np.array(matrix)

    std = (matrix - matrix.mean(axis=0)) / \
        (matrix.max(axis=0) - matrix.min(axis=0))

    return std
