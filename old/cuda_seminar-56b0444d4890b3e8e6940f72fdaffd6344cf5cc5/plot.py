from matplotlib import pyplot as plt
import numpy as np

data = np.genfromtxt("out.txt")

plt.plot(data[:,0], data[:,1])
plt.plot(data[:,0], data[:,2])

plt.show()
