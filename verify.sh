#!/bin/bash

# Horrible script to just count output files etc
# Definitely a work in progress

alg=$1
dornd=$2

EXP_DS=6000
RESTARTS=50

if [ $dornd == "nd" ]; then
    echo "ND"
    EXP_FILES=$((EXP_DS*RESTARTS))
else
    EXP_FILES=$EXP_DS
fi


echo "ALGORITHM: $alg"

numlabels=`find "./_output/$alg/" -name "labels*.csv" | wc -l`
echo "LABELS FILES: $numlabels/$EXP_FILES"

numoutputs=`find "./_output/$alg/" -name "output*.csv" | wc -l`
echo "OUTPUT FILES: $numoutputs/$EXP_FILES"

echo "Synthetic datasets..."

numdatasets=0
for folder in `ls _output/$alg/synthetic/`; do

    counted=`ls _output/$alg/synthetic/$folder | wc -l`

    echo "In $folder:"

    echo -e "\twe found $counted dataset folders"
    numdatasets=$(($numdatasets+$counted))

    numlabels=`find "./_output/$alg/synthetic/$folder" -name "labels*.csv" | wc -l`
    echo -e "\twe found $numlabels label files"

    numout=`find "./_output/$alg/synthetic/$folder" -name "output*.csv" | wc -l`
    echo -e "\twe found $numout output files"

done

echo "TOTAL datasets: $numdatasets/$EXP_DS"

