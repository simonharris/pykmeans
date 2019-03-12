#
# Utility functions shamelessly taken from my work on the CE705 assignments
#
import csv
import random
import statistics

def get_standard_deviation(matrix, index):
    '''Calculates corrected sample standard deviation for a column of a matrix'''

    '''Note: I'd be tempted to use statistics.stdev() but the question
    seems to imply it's desirable to show our working. I may have
    misinterpreted it, of course. Either way, the unit tests show that
    I get the same results as statistics.stdev()'''

    # isolate the column we need
    column = [row[index] for row in matrix]

    # required for the calculation
    N = len(column)
    mean = sum(column)/N

    total = 0
    for element in column:
        total += ((element - mean)**2)

    return (total/(N-1))**0.5


def get_standardised_matrix(matrix):
    '''Computes a standardised version of a matrix i.e. mean=0, variance=1'''

    numrows = len(matrix)
    numcolumns = len(matrix[0])

    # collect each column and its standard deviation and mean in a dict
    columns = [None] * numcolumns

    index = 0
    while index < numcolumns:

        columninfo = {'elements':[], 'mean':None, 'deviation':None}

        columninfo['elements'] = [row[index] for row in matrix]
        columninfo['mean'] = sum(columninfo['elements']) / len(columninfo['elements'])
        columninfo['deviation'] = get_standard_deviation(matrix, index)
        columns[index] = columninfo
        index += 1

    # start to build the standardised matrix - copy() gets us the desired dimensions
    output = matrix.copy()

    rowindex = 0
    while rowindex < numrows:

        columnindex = 0
        while columnindex < numcolumns:
            output[rowindex][columnindex] = (
                (matrix[rowindex][columnindex] - columns[columnindex]['mean'])
                /
                columns[columnindex]['deviation'])
            columnindex += 1

        rowindex += 1

    return output
