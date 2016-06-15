from PIL import Image
import numpy as np
import string, sys, math


in_fp = file(sys.argv[1], 'r')
all_file = in_fp.readlines()
in_fp.close()

elements = string.split(all_file[1])
print (len(all_file), len(elements))
w, h = len(all_file), len(elements)
data = np.zeros((h, w, 3), dtype=np.uint8)

for i in range(len(all_file)):
	elements = string.split(all_file[i])
	for j in range(len(elements)):
		if (elements[j] == "nan"):
			temp = 255.
		elif (float(elements[j]) > 255.):
			temp = 255.
		elif (float(elements[j]) < 0.):
			temp = 0.
		else:		
			temp = float(elements[j])
		data[i][j] = int(temp)

img = Image.fromarray(data, 'RGB')
img.save('temperature.png')
