'''Fetcher for the Wisconsin Breast Cancer (Original) dataset'''

import numpy as np
import pandas as pd


#URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'

URL = 'breast-cancer-wisconsin.data.csv'


names = [
    'SCN',                         # id number
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


ds = pd.read_csv(URL, names=names, na_values=['?'])

# Lose the ID column
ds = ds.drop('SCN', axis=1)

# Drop rows with missing data
ds = ds.dropna()

labels = ds.loc[:,'Class'].values

labels = labels > 3

labels = labels.astype('int')


np.savetxt('labels.csv', labels)		




