import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


def hyper_gen(y):
        x = np.sqrt(1+y**2)
        return x

def parab_gen(y,a):
        x = y**2/a
        return x

#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 100
y = np.linspace(-5,5,len)



#Generating the Standard Hyperbola                                                                                                           
x = hyper_gen(y)
xStandardHyperLeft = np.vstack((-x,y))
xStandardHyperRight = np.vstack((x,y))

V = np.array(([0,1.5],[1.5,2]))
u = np.array(([0,0.5]))
f =  -6
Vinv = LA.inv(V)
#Eigenvalues and eigenvectors                                                                                                                 
D_vec,P = LA.eig(V)
D = np.diag(D_vec)
#print(D,P)                                                                                                                                   
uconst = u.T@Vinv@u-f
a = np.sqrt(np.abs(uconst/D_vec[0]))
b = np.sqrt(np.abs(uconst/D_vec[1]))

#Affine Parameters                                                                                                                            
c = -Vinv@u
#print(c)                                                                                                                                     
R =  np.array(([0,1],[1,0]))
ParamMatrix = np.array(([a,0],[0,b]))

#Generating the eigen hyperbola                                                                                                               
xeigenHyperLeft = R@ParamMatrix@xStandardHyperLeft
xeigenHyperRight = R@ParamMatrix@xStandardHyperRight

#Generating the actual hyperbola                                                                                                              
xActualHyperLeft = P@ParamMatrix@R@xStandardHyperLeft+c[:,np.newaxis]
xActualHyperRight = P@ParamMatrix@R@xStandardHyperRight+c[:,np.newaxis]

#Plotting the actual hyperbola                                                                                                                
plt.plot(xActualHyperLeft[1,:],xActualHyperLeft[0,:],label='1.x+2.x^2+3.xy+0.y=6',color='b')
plt.plot(xActualHyperRight[1,:],xActualHyperRight[0,:],color='b')

#hyper parameters
V = np.array(([0,1.5],[1.5,1]))
u = np.array(([0.5,1]))
f =  -5
Vinv = LA.inv(V)
#Eigenvalues and eigenvectors
D_vec,P = LA.eig(V)
D = np.diag(D_vec)
#print(D,P)
uconst = u.T@Vinv@u-f
a = np.sqrt(np.abs(uconst/D_vec[0]))
b = np.sqrt(np.abs(uconst/D_vec[1]))

#Affine Parameters
c = -Vinv@u
#print(c)
R =  np.array(([0,1],[1,0]))
ParamMatrix = np.array(([a,0],[0,b]))

#Generating the eigen hyperbola
xeigenHyperLeft = R@ParamMatrix@xStandardHyperLeft
xeigenHyperRight = R@ParamMatrix@xStandardHyperRight

#Generating the actual hyperbola
xActualHyperLeft = P@ParamMatrix@R@xStandardHyperLeft+c[:,np.newaxis]
xActualHyperRight = P@ParamMatrix@R@xStandardHyperRight+c[:,np.newaxis]

#Plotting the actual hyperbola
plt.plot(xActualHyperLeft[1,:],xActualHyperLeft[0,:],label='2.x+1.x^2+3.xy+1.y=5',color='r')
plt.plot(xActualHyperRight[1,:],xActualHyperRight[0,:],color='r')
#

y = np.linspace(-3,3,len)
V = np.array(([0,0],[0,1]))
u = np.array(([-0.5,-0.5]))
f = 7

O = np.array(([0,0]))
#Generating the Standard parabola                                                                                                            \
                                                                                                                                              

#Eigenvalues and eigenvectors                                                                                                                \
                                                                                                                                              
D_vec,P = LA.eig(V)
D = np.diag(D_vec)

p = P[:,0]
eta = 2*u@p

foc = eta/D_vec[1]

x = parab_gen(y,foc)

cA = np.vstack((u+eta*0.5*p,V))
cb = np.vstack((-f,(eta*0.5*p-u).reshape(-1,1)))
c = LA.lstsq(cA,cb,rcond=None)[0]
c = c.flatten()
print(c,foc)
P=-P

xStandardparab = np.vstack((x,y))

xActualparab = P@xStandardparab + c[:,np.newaxis]

plt.plot(xActualparab[1,:],xActualparab[0,:],label='1.x-1.x^2+0.xy+1.y=7',color='g')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()
