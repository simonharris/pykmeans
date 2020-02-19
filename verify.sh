#!/bin/bash

# Horrible script to just count output files etc
# Definitely a work in progress

alg=$1
dornd=$2

RESTARTS=50

echo -e "ALGORITHM: $alg\n"

## -----------------------------------------------------------------------------

EXP_DS=6000

if [ $dornd == "nd" ]; then
    EXP_FILES=$((EXP_DS*RESTARTS))
else
    EXP_FILES=$EXP_DS
fi

echo "SYNTHETIC DATASETS..."
numlabels=`find "./_output/$alg/synthetic/" -name "labels*.csv" | wc -l`
echo "Label files: $numlabels/$EXP_FILES"

numoutputs=`find "./_output/$alg/synthetic/" -name "output*.csv" | wc -l`
echo "Output files: $numoutputs/$EXP_FILES"

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
    
    numerror=`find "./_output/$alg/synthetic/$folder" -name "exception*.csv" | wc -l`
    echo -e "\twe found $numerror exception files"
done

echo -e "TOTAL datasets: $numdatasets/$EXP_DS\n"

## -----------------------------------------------------------------------------



EXP_DS=28

if [ $dornd == "nd" ]; then
    EXP_FILES=$((EXP_DS*RESTARTS))
else
    EXP_FILES=$EXP_DS
fi

MYDIR="./_output/$alg/realworld/"

echo "REAL-WORLD DATASETS..."

if [ -d $MYDIR ]; then

    numlabels=`find $MYDIR -name "labels*.csv" | wc -l`
    echo "Label files: $numlabels/$EXP_FILES"

    numoutputs=`find $MYDIR -name "output*.csv" | wc -l`
    echo "Output files: $numoutputs/$EXP_FILES"

    counted=`ls $MYDIR | wc -l`

    echo "In $MYDIR:"

    echo -e "\twe found $counted dataset folders"

    numlabels=`find $MYDIR -name "labels*.csv" | wc -l`
    echo -e "\twe found $numlabels label files"

    numout=`find $MYDIR -name "output*.csv" | wc -l`
    echo -e "\twe found $numout output files"

    numerror=`find $MYDIR -name "exception*.csv" | wc -l`
    echo -e "\twe found $numerror exception files"

    echo "TOTAL datasets: $counted/$EXP_DS"
    
 else
    echo "No directory $MYDIR"
 fi
