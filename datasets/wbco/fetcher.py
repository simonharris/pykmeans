'''Fetcher for the Wisconsin Breast Cancer (Original) dataset'''

import numpy as np
import pandas as pd


## Config ----------------------------------------------------------------------


URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'

INT_FMT = '%1i'

names = [
    'SCN',                          # id number
    'Clump Thickness',              # 1 - 10
    'Uniformity of Cell Size',      # 1 - 10
    'Uniformity of Cell Shape',     # 1 - 10
    'Marginal Adhesion',            # 1 - 10
    'Single Epithelial Cell Size',  # 1 - 10
    'Bare Nuclei',                  # 1 - 10
    'Bland Chromatin',              # 1 - 10
    'Normal Nucleoli',              # 1 - 10
    'Mitoses',                      # 1 - 10
    'Class',                        # (2 for benign, 4 for malignant)
]


## Run -------------------------------------------------------------------------


# Read data from UCI
ds = pd.read_csv(URL, names=names, na_values=['?'])

# Lose the ID column
ds = ds.drop('SCN', axis=1)

# Drop rows with missing data
ds = ds.dropna()

# Convert the 2/4 labels to 1/0 and move to another dataframe
labels = ds.loc[:,'Class'].values > 3
ds = ds.drop('Class', axis=1)

# Save files
np.savetxt('labels.csv', labels, fmt=INT_FMT, delimiter=',')
np.savetxt('data.csv', ds, fmt=INT_FMT, delimiter=',')
		
