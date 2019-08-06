'''Fetcher for the Soybean (Small) dataset'''

import numpy as np
import pandas as pd

## Config ----------------------------------------------------------------------

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data'

INT_FMT = '%1i'

class_map = ['D1', 'D2', 'D3', 'D4']

## Run -------------------------------------------------------------------------

# Read data from UCI
ds = pd.read_csv(URL, header=None)

# Convert the Dn labels to 0..3 and move to another dataframe
labels = np.searchsorted(class_map, ds.loc[:,35].values)
ds = ds.drop(35, axis=1)

# Save files
np.savetxt('labels.csv', labels, fmt=INT_FMT)
np.savetxt('data.csv', ds, fmt=INT_FMT, delimiter=',')
		
