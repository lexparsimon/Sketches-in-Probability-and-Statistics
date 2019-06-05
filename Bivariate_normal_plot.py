#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
cov = np.array([[1., 0.6],
       [0.6, 1.]])

# random seed for reproducibility
np.random.seed(543212345)

#Bivariate Distribution of X and Y
x, y = np.random.multivariate_normal(mean, cov, 7500).T

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
space = 0.005

scatter_frame = [left, bottom, width, height]
x_histogram = [left, bottom + height + space, width, 0.2]
y_histogram = [left + width + space, bottom, 0.2, height]

# start Figure
fig = plt.figure(figsize=(8, 8))

scatter_plot = plt.axes(scatter_frame)
scatter_plot.tick_params(direction='in', top=True, right=True)
ax_histx = plt.axes(x_histogram)
ax_histx.tick_params(direction='in', labelbottom=False)
ax_histy = plt.axes(y_histogram)
ax_histy.tick_params(direction='in', labelleft=False)

# defining plot limits:
binwidth = 0.25
lim = np.ceil((np.abs([x, y]).max() + 1) / binwidth) * binwidth
scatter_plot.set_xlim((-lim, lim))
scatter_plot.set_ylim((-lim, lim))

bins = np.arange(-lim, lim + binwidth, binwidth)
ax_histx.hist(x, bins=bins, alpha=0.5)
ax_histy.hist(y, bins=bins, orientation='horizontal', alpha=0.5)

ax_histx.set_xlim(scatter_plot.get_xlim())
ax_histy.set_ylim(scatter_plot.get_ylim())

scatter_plot.scatter(x, y, s=0.5, alpha=0.75)
scatter_plot.set_xlabel(r"$X$", color='k', rotation='horizontal')
scatter_plot.yaxis.set_label_coords(0.5,-0.15)
scatter_plot.set_ylabel(r"$Y$", color='k', rotation='horizontal')
scatter_plot.yaxis.set_label_coords(-0.075,0.5)

plt.show()


# In[ ]:




