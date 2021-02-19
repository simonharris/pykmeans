#!/bin/bash

# ugly script to munge together some larger datasets

srcdir="10_50_1000_u_1_010"
targetdira="10_50_2000_u_1_010"
targetdirb="10_50_10000_u_1_010"
targetdirc="10_50_50000_u_1_010"


if [ ! -d $targetdira ]
then
	mkdir $targetdira
fi
if [ ! -d $targetdirb ]
then
	mkdir $targetdirb
fi
if [ ! -d $targetdirc ]
then
	mkdir $targetdirc
fi	
		
rm -f $targetdira/*
rm -f $targetdirb/*
rm -f $targetdirc/*


# times by 2 = 2000
touch $targetdira/labels.csv
touch $targetdira/data.csv

for i in {1..2};
do
	# echo "Looping $i "
	cat $srcdir/labels.csv >> $targetdira/labels.csv
	cat $srcdir/data.csv >> $targetdira/data.csv
done


# times by 10 = 10000
touch $targetdirb/labels.csv
touch $targetdirb/data.csv

for i in {1..10};
do
	# echo "Looping $i "
	cat $srcdir/labels.csv >> $targetdirb/labels.csv
	cat $srcdir/data.csv >> $targetdirb/data.csv
done


# times by 50 = 50000
touch $targetdirc/labels.csv
touch $targetdirc/data.csv

for i in {1..50};
do
	# echo "Looping $i "
	cat $srcdir/labels.csv >> $targetdirc/labels.csv
	cat $srcdir/data.csv >> $targetdirc/data.csv
done


wc -l $targetdira/*
wc -l $targetdirb/*
wc -l $targetdirc/*

