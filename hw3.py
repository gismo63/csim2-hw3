import numpy as np
from scipy import linalg

def steepest_descent(A,b,x):
    r = b - np.dot(A,x)
    r_squared = np.dot(r,r)
    i = 1
    while r_squared > 1e-24:
        A_r = np.dot(A,r)
        alpha = (r_squared)/(np.dot(r,A_r))
        x += alpha*r
        if i%10==0:
            r = b - np.dot(A,x)
        else:
            r -= alpha*A_r
        r_squared = np.dot(r,r)
        i+=1
    print (i)
    return x

def cgd(A,b,x):
    r = b - np.dot(A,x)
    d = r
    r_squared = np.dot(r,r)
    i = 0
    while r_squared > 1e-24:
        A_d = np.dot(A,d)
        alpha = (r_squared)/(np.dot(d,A_d))
        x += alpha*d
        r_old = r
        r_squared_old = r_squared
        r -= alpha*A_d
        r_squared = np.dot(r,r)
        beta = r_squared/r_squared_old
        d = r + beta*d
        i+=1
    print (i)
    return x


    return x




A = np.loadtxt("A_matrix.txt")
b = np.array([1.,4.3,2.,-5.,0.,1.,6.,2.,-9.4,10.,0.,-1.,7.])
x = np.zeros(13)

x = steepest_descent(A,b,x)
print (x)
print (np.dot(A,x)-b,"\n")
x = np.zeros(13)
x = cgd(A,b,x)
print (x)
print (np.dot(A,x)-b,"\n")
