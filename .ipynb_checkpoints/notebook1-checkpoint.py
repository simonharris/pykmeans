{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# pycluster"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Import statements"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sklearn.datasets as skdatasets\n",
    "import sklearn.cluster as skcluster\n",
    "import sklearn.metrics as skmetrics\n",
    "import kmeans\n",
    "import utils\n",
    "from initialisations import random, ikmeans\n",
    "import sys\n",
    "from argparse import ArgumentParser"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "datasets = {\n",
    "    'iris':  skdatasets.load_iris,\n",
    "    'wine':  skdatasets.load_wine,\n",
    "    'bc':    skdatasets.load_breast_cancer,\n",
    "}\n",
    "\n",
    "algorithms = {\n",
    "    'random': random.generate,\n",
    "    'ikmeans': ikmeans.generate,\n",
    "}"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
