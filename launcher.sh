#!/bin/bash

for i in `seq 1 100`;
	do
	./sim2d dispersion/ -l 1000 -bn 16 -n 2 -nographics 1100
	done

exit
