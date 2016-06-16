import numpy as np
import string, sys, math

root = "time-8-8-cpu.txt"
name = [root] 

out_file = open("cpu.txt","w")
k = 2

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
