import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time

def steepest_descent(A,b,x):
    errs = []
    r = b - np.dot(A,x)
    r_squared = np.dot(r,r)
    i = 1
    while r_squared > 1e-24:
        errs.append(r_squared)
        A_r = np.dot(A,r)
        alpha = (r_squared)/(np.dot(r,A_r))
        x += alpha*r
        if i%10==0:
            r = b - np.dot(A,x)
        else:
            r -= alpha*A_r
        r_squared = np.dot(r,r)
        i+=1
    print ("Steepest Decent: ")
    print ("# Iterations: ",i)
    return x, errs

def cgd(A,b,x):
    errs = []
    r = b - np.dot(A,x)
    d = r
    r_squared = np.dot(r,r)
    i = 0
    while r_squared > 1e-24:
        errs.append(r_squared)
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
    print ("Conjugate Gradient Decent:")
    print ("# Iterations: ",i)
    return x, errs



np.random.seed(10)
A = np.loadtxt("A_matrix.txt")
b = np.random.normal(scale = 10,size = 13)
x = np.random.normal(scale = 10,size = 13)

start = time.clock()
x, errs_sd = steepest_descent(A,b,x)
end = time.clock()
print("Convergence Time: ", (end - start)*1000, "ms")
print ("x")
print (x)
print ("\n")
print ("Ax-b")
print (np.dot(A,x)-b,"\n")
x = np.zeros(13)

start = time.clock()
x, errs_cgd = cgd(A,b,x)
end = time.clock()
print("Convergence Time: ", (end - start)*1000, "ms")
print ("x")
print (x)
print ("\n")
print ("Ax-b")
print (np.dot(A,x)-b,"\n")


plt.ylabel("$r^2$")
plt.xlabel("# Iterations")
plt.yscale('log')

plt.plot(errs_sd, label = "Steepest Descent")


plt.yscale('log')

plt.plot(errs_cgd, label = "Conjugate Gradient Descent")
plt.legend()
plt.show()
