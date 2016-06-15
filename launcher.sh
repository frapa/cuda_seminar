#!/bin/bash

./sim2d dispersion/ -l 300 -nb 8 -n 4
./sim2d dispersion/ -l 300 -nb 8 -n 8
./sim2d dispersion/ -l 300 -nb 8 -n 16
./sim2d dispersion/ -l 300 -nb 8 -n 32
./sim2d dispersion/ -l 300 -nb 8 -n 64

exit
