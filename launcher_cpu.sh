#!/bin/bash

for i in `seq 1 100`;
	do
	./sim2d dispersion1024/ -l 1000 -bn 16 -n 4 -nographics 1100
	./sim2d dispersion1024/ -l 1000 -cpu 1100
	done

exit
