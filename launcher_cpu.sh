#!/bin/bash

for i in `seq 1 100`;
	do
	./sim2d dispersion256/ -l 100 -bn 16 -n 4 -nographics 110
	./sim2d dispersion256/ -l 100 -cpu 110
	done

exit
