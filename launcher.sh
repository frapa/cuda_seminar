#!/bin/bash

for i in `seq 1 100`;
	do
	./sim2d dispersion/ -l 1000 -bn 4 -n 16 -nographics 1100
	done
for i in `seq 1 100`;
	do
	./sim2d dispersion/ -l 1000 -bn 4 -n 32 -nographics 1100
	done
for i in `seq 1 100`;
	do
	./sim2d dispersion/ -l 1000 -bn 4 -n 64 -nographics 1100
	done
for i in `seq 1 100`;
	do
	./sim2d dispersion/ -l 1000 -bn 4 -n 128 -nographics 1100
	done

exit
