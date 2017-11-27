import os, sys
import numpy as np
import cv2 as cv

import scipy.io as io


def project_points(X, K, R, T, distortion_flag=False, distortion_params=None):
        """
        Your implementation goes here!
        """
        # Project points from 3d world coordinates to 2d image coordinates
        return X_camera

def project_and_draw(img, X_3d, K, R, T, distortion_flag, distortion_parameters):
	"""
		Your implementation goes here!
	"""
	# call your "project_points" function to project 3D points to camera coordinates
	# draw the projected points on the image and save your output image here
	# cv.imwrite(output_name, img_array)

	return True

if __name__ == '__main__':
	base_folder = './data/'

	image_num = 1
	data = io.loadmat('./data/ex1.mat')
	X_3D = data['X_3D'][0]
	TVecs = data['TVecs']		# Translation vector: as the world origin is seen from the camera coordinates
	RMats = data['RMats']		# Rotation matrices: converts coordinates from world to camera
	kc = data['dist_params']	# Distortion parameters
	Kintr = data['intinsic_matrix']	# K matrix of the cameras
	
	imgs = [cv.imread(base_folder+str(i).zfill(5)+'.jpg') for i in range(TVecs.shape[0])]

	project_and_draw(imgs[image_num], X_3D, Kintr, RMats[image_num], TVecs[image_num], kc)
