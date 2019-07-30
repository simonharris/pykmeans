# pycluster

Experiments in initialisation strategies for the K-means data clustering algorithm, as research for my MSc by Dissertation at the University of Essex.

## Structure

 - `initialisations/`: implementations of K-means initialisation algorithms
 - `datasets/ `: data importers, preprocessors/wranglers and resulting data
 - `metrics/`: implementations and wrappers of algorithms used to measure clustering success
 - `notebooks/`: Jupyter notebooks used to demonstrate clustering using the initialisations
 - `tests/`: unit tests
 - `kmeans.py`: implementation of K-means algorithm. It is not anticipated that this will be used for the experiments, though parts of it have been incorporated into implemented initialisations 
