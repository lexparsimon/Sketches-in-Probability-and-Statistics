#!/usr/bin/env python
# coding: utf-8

# In[4]:


#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
from scipy.stats import mvn
from numpy.linalg import inv
import matplotlib.patches as mpatches

# setting mean and covariance 
mean = np.array([0., 0.])

# Here we define phi as a function of 
def phi_func(rho, K):
    
    '''Here we define the phi as a function of rho and K'''
    
    # construct an array of covariance matrices for each rho
    COV = np.array([[
            [1, r],
            [r, 1]] for r in rho]) 
    
    # scipy doesn't offer a survival function (i.e. complementary cdf), so we have to build it  
    threshold = np.array([K, K])
    upper = np.array([100, 100])
    nom_phi = np.array([mvn.mvnun(threshold,upper,mean,cov)[0] for cov in COV])
    
    return nom_phi/(1 - norm.cdf(K)) 


# In[ ]:




