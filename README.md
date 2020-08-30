# pycluster

Experiments in initialisation strategies for the K-means data clustering algorithm, as research for my MSc by Dissertation at the University of Essex.

## Structure

 - `runner.py`: the main bootstrap file. Runs experiments using parallelisation where possible
 - `initialisations/`: implementations of K-means initialisation algorithms
 - `datasets/ `: data importers, preprocessors/wranglers and resulting data
 - `metrics/`: implementations and wrappers of algorithms used to measure clustering success
 - `notebooks/`: Jupyter notebooks used to demonstrate clustering using the initialisations
 - `tests/`: unit tests
 - `kmeans.py`: implementation of K-means algorithm. It is not anticipated that this will be used for the experiments, though parts of it have been incorporated into implemented initialisations 
 - `cluster.py`: ...
 - `dataset.py`: ...

## Usage

``$   python3 runner.py <algorithm> <datadir> <restarts>``
 
The parameters which must be supplied to `runner.py` as above are:
 - `algorithm`: the identifier for the initialisation algorithm to be run, each of which can be found in Table 4.12
 - `datadir`: the relative path to the directory containing the data sets for the experimental run
 - `restarts`: the  number  of  restarts  to  be  performed  per  data  set,  which will typically be 1 for deterministic initialisation algorithms and more for non-deterministic algorithms


