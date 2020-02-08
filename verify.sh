#!/bin/bash

# Horrible script to just count output files etc
# Definitely a work in progress


alg=$1
dornd=$2

echo "ALG:" $alg

numlabels=`find "./_output/$alg/" -name "labels*.csv" | wc -l`
echo "LABELS FILES:" $numlabels

numoutputs=`find "./_output/$alg/" -name "output*.csv" | wc -l`
echo "OUTPUT FILES:" $numoutputs

echo "Synthetic datasets..."

numdatasets=0 
for folder in `ls _output/$alg/synthetic/`; do

    counted=`ls _output/$alg/synthetic/$folder | wc -l`

    echo "In" $folder "we found" $counted
    numdatasets=$(($numdatasets+$counted))
done

echo "TOTAL datasets:" $numdatasets

