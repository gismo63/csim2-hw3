import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time


def steepest_descent(A,b,x):
    errs = [] #This list will store the errors so they can be graphed
    r = b - np.dot(A,x)
    r_squared = np.dot(r,r)
    i = 0
    while r_squared > 1e-24:
        errs.append(r_squared)
        A_r = np.dot(A,r) #pre-calculate A*r because it is used twice
        alpha = (r_squared)/(np.dot(r,A_r))
        x += alpha*r
        if i%10==0: #One in every 10 times calculate r exactly
            r = b - np.dot(A,x)
        else: #This method of calculating r is faster but is more likely to create rounding errors
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
        A_d = np.dot(A,d) #pre-calculate A*d because it is used twice
        alpha = (r_squared)/(np.dot(d,A_d))
        x += alpha*d
        r_squared_old = r_squared #need r_i and r_{i+1} to calculate beta
        r -= alpha*A_d
        r_squared = np.dot(r,r)
        beta = r_squared/r_squared_old
        d = r + beta*d
        i+=1
    print ("Conjugate Gradient Decent:")
    print ("# Iterations: ",i)
    return x, errs



np.random.seed(10) #x_0 and b are randomly generated, this seed is used for reproducibility
A = np.loadtxt("A_matrix.txt")
b = np.random.normal(scale = 10,size = 13)
x = np.random.normal(scale = 10,size = 13)

start = time.clock()
x, errs_sd = steepest_descent(A,b,x)
end = time.clock()
slope_sd, intercept_sd = np.polyfit(np.arange(len(errs_sd)),np.log10(errs_sd),1) # linear fit to the log of the data
print("Convergence Time: ", (end - start)*1000, "ms")
print ("x")
print (x)
print ("\n")
print ("Ax-b")
print (np.dot(A,x)-b,"\n")
print ("Error Squared Scales as: 10^(", slope_sd, "*#iterations)\n\n" )
x = np.zeros(13)

start = time.clock()
x, errs_cgd = cgd(A,b,x)
end = time.clock()
slope_cgd, intercept_cgd = np.polyfit(np.arange(len(errs_cgd)),np.log10(errs_cgd),1)
print("Convergence Time: ", (end - start)*1000, "ms")
print ("x")
print (x)
print ("\n")
print ("Ax-b")
print (np.dot(A,x)-b,"\n")
print ("Error Squared Scales as: 10^(", slope_cgd, "*#iterations)" )




plt.ylabel("$r^2$")
plt.xlabel("# Iterations")
plt.yscale('log')

plt.plot(errs_sd, label = "Steepest Descent")
plt.plot(np.arange(len(errs_sd)), 10**(slope_sd*np.arange(len(errs_sd))+intercept_sd),label = "Steepest Descent Linear Fit")


plt.yscale('log')

plt.plot(errs_cgd, label = "Conjugate Gradient Descent")
plt.plot(np.arange(len(errs_cgd)), 10**(slope_cgd*np.arange(len(errs_cgd))+intercept_cgd),label = "Conjugate Gradient Descent Linear Fit")
plt.legend()
plt.show()
