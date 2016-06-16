import numpy as np
import string, sys, math

root = "time-32"
name = [root+"-"+str(i)+".txt" for i in [1, 2, 4, 8, 16]] 

out_file = open("32blocks.txt","w")
k = 1

for i in name:
	in_fp = file(i, 'r')
	all_file = in_fp.readlines()
	in_fp.close()

	data = [0 for i in range(len(all_file))]
	for i in range(len(all_file)):
		elements = string.split(all_file[i])
		data[i] = float(elements[1])
		
	mean = np.mean(data)
	dvst = np.std(data)
	
	out_file.write(str(k) + "    " + str(mean) + "   " + str(dvst) + "\n")
	k *= 2

out_file.close()
