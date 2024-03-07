## MPHil in data intensive science
# Astronomy in the SKA-era mini project
# SKA-low mini project
# 01.03.2024
# Q. Gueuning (qdg20@cam.ac.uk) and O. O'Hara

import numpy as np
from harp_beam import compute_EEPs
import scipy.io

## Q 2. plot all the 256 EEPs and their average (AEP)
num_dir = 2  # number of samples along theta
theta = np.array([0.0, np.pi/4.0])[:,None]  # angle from zenith in radians
phi = np.array([np.pi/2.0, 0.0])[:,None]   # azimuth from x-axis in radians (from 0 to 2pi)
# !! theta and phi must be vectors of the same size

# antenna positions loaded as follows
data_folder = 'harp_beam'
filename_eep = f"data_EEPs_SKALA41_random_100MHz.mat"
mat = scipy.io.loadmat(filename_eep)
pos_ant = np.array(mat['pos_ant'])
x_pos = pos_ant[:,0]
y_pos = pos_ant[:,1]

# the following function computes the theta and phi vector components of the 256 embedded element patterns
# evaluated at coordinates "theta" and "phi" defined earlier by the user
v_theta_polY, v_phi_polY, v_theta_polX, v_phi_polX = compute_EEPs(theta, phi)

# these voltage matrices are of size num_theta*num_phi x num_ant
# The EEP of antenna i_ant at position pos_ant(i_ant,:) is in v[:, i_ant]

## Q 3. the model matrix, the covariance matrix, the exact gain values and (my) gain estimations are loaded as
filename_vismat = f"data_20feb2024_2330_100MHz.mat"
mat = scipy.io.loadmat(filename_vismat)
R = np.array(mat['R']) # covariance matrix
M_AEP = np.array(mat['M_AEP']) # model matrix using AEP
M_EEPs = np.array(mat['M_EEPs']) # model matrix using all EEPs
g_sol = np.array(mat['g_sol']) # exact gain solution
g_AEP = np.array(mat['g_AEP']) # estimation using M_AEP (using this for question 5 and 6 only if you haven't been able to complete question 3 and 4)
g_EEPs = np.array(mat['g_EEPs']) # estimation using M_EEPs