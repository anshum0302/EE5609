import numpy as np
import matplotlib.pyplot as plt

# theta goes from 0 to 2pi
theta = np.linspace(0, 2*np.pi, 100)

# the radius of the circle
r = np.sqrt(2)/2
# compute x1 and x2
x = -4+r*np.cos(theta)
y = 1+r*np.sin(theta)

# create the figure
plt.plot(x,y)
plt.axis('equal')
plt.text(-4,1,'. (-4,1)',fontsize=10,color='red')
plt.hlines(y=0, xmin=-5.5,xmax=0.5,color='black')
plt.vlines(x=0, ymin=-1,ymax=3,color='black')
#lt.show()
plt.savefig('circle.png')
