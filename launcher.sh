#!/bin/bash

./sim2d dispersion/ -l 1000 -nb 32 -n 1 -nographics 1100
./sim2d dispersion/ -l 1000 -nb 32 -n 2 -nographics 1100
./sim2d dispersion/ -l 1000 -nb 32 -n 4 -nographics 1100
./sim2d dispersion/ -l 1000 -nb 32 -n 8 -nographics 1100
./sim2d dispersion/ -l 1000 -nb 32 -n 16 -nographics 1100

exit
