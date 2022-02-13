# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:44:30 2019

@author: Yichen Jiang
"""

import numpy as np

import bspline
import bspline.splinelab as splinelab
import matplotlib.pyplot as plt
import pandas as pd 
import os 

# In[]
 
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'

# In[]
"""
--- define knots / knot vector ---

"""

list_year = [0,
 23.5,
 112.5,
 366.5,
 768.5,
 1293.5,
 1800.5,
 2223.5,
 2588.5,
 2958.5,
 3332.5,
 3772.5,
 4179.5,
 4533]

# In[]
"""
--- parameter initialization ---

"""

p = 3              # order of spline (as-is; 3 = cubic)
nknots = 12       # number of knots to generate (here endpoints count only once)
tau = np.arange(0,4534)  # collocation sites (i.e. where to evaluate)

knots = list_year                   # create a knot vector without endpoint repeats
k     = splinelab.augknt(knots, p)  # add endpoint repeats as appropriate for spline order p
B     = bspline.Bspline(k, p)       # create spline basis of order p on knots k

A0 = B.collmat(tau)                 # collocation matrix for function value at sites tau

# In[]
"""
--- plot b spline basis function ---

"""

plt.figure(dpi = 40, figsize = (30, 20))
index = np.linspace(0,len(A0)-1,len(A0))
for i in range(len(A0.T)):
    plt.plot(index,A0.T[i], color='k')
plt.show()

# In[]:
"""
--- revise columns name ---

"""
df_bspline = pd.DataFrame(A0)
dict_column = {}

for column in df_bspline.columns:
    if column not in dict_column.keys():
        dict_column[column] = 'b_spline_'+str(column)

df_bspline = df_bspline.rename(columns=dict_column)

# In[]
"""
--- export b spline basis function data ---

"""

df_bspline.to_csv(os.path.join(path,'b spline basis function.csv'), index=False, quoting=1)



