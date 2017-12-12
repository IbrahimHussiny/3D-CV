import os, sys
import numpy as np
from numpy.linalg import inv
import cv2 as cv
import cmath

import scipy.io as io

if __name__ == '__main__':
    data = io.loadmat('./ex2.mat')
    print data.keys()

    #read instrinsic paramters
    alpha_x = data['alpha_x'][0][0]
    alpha_y = data['alpha_y'][0][0]
    x_0 = data['x_0'][0][0]
    y_0 = data['y_0'][0][0]
    s = data['s'][0][0]

    #read H matrices
    H1 = data['H1']
    H2 = data['H2']
    H3 = data['H3']

    # build the K matrix 
    K = np.matrix([[alpha_x, s, x_0], [0 ,alpha_y, y_0], [0, 0, 1]])
    print K

    #since we assume t is [0 0 0] we have H = KR(K^-1) -> R = (K^-1)HK
    R1 = np.dot(np.dot(inv(K),H1),K)
    R2 = np.dot(np.dot(inv(K),H2),K)
    print R1,R2
