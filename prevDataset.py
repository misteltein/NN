import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt( 'spiral3.csv', delimiter = ',' )
fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(data[:,0],data[:,1],c=data[:,2],s=1,alpha=0.5)
plt.show()
