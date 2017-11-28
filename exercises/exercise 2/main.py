import os, sys
import numpy as np
import cv2 as cv

import scipy.io as io

outputFolderPath = "./outputs/"

def getExtrinsicParametersMat(RotationMatrix, T):
	# upper left 3*3 matrix 
	extrinsicParametersMat = np.array(RotationMatrix)
	extrinsicParametersMat.reshape(3,3)

	#-RC (upper right matrix of the extrinsic parameters mat)
	#RC = np.matrix(-1 * np.dot(extrinsicParametersMat , np.array(T)) )
	RC = np.matrix(  np.array(T) )
	extrinsicParametersMat = np.concatenate((extrinsicParametersMat, RC), axis =1)
	
	# add last row of the matrix [0,0,0,1]
	extrinsicParametersMat = np.vstack([extrinsicParametersMat, np.array([0,0,0,1])])
	
	return extrinsicParametersMat

def getIntrinsicParametersMat(K):
	#Intrinsic matrix = K[I3 | 03]
	return np.dot(np.array(K.reshape(3,3)),np.append(np.identity(3),np.zeros((3,1)), axis =1))

# Project points from 3d world coordinates to 2d image coordinates
def project_points(X, K, R, T, distortion_flag=False, distortion_params=None):
	# 2D points in pixel coordinates mapped from the given 3D points 
	cameraPoints = []

	#construct the P matrix	[Intrinsic * Extrinisc]
	P = np.dot(getIntrinsicParametersMat(K), getExtrinsicParametersMat(R,T))
	for i in range(0,len(X[0])):
		#world point in homogeneous coordinates 
		worldPoint = np.array([X[0][i],X[1][i],X[2][i],1])
		#corresponding point in pixel cooridnates 
		pixelPoint = np.dot(P, worldPoint)
		pixelPoint = np.transpose(pixelPoint)
		#recovering homogeneous coordinates by deviding over the last element, and add it to results
		cameraPoints.append(pixelPoint[:-1]/pixelPoint[-1])

	return cameraPoints
def project_and_draw(img, X_3d, K, R, T, image_name, distortion_flag, distortion_parameters):
	
	projectedPoints = project_points(X_3d, K, R,T,distortion_flag,distortion_parameters)
	for projectedPoint in projectedPoints:
		cv.circle(img, tuple(projectedPoint), 1, (0,255,0),-1)
	cv.imwrite(outputFolderPath+image_name, img)

if __name__ == '__main__':
	base_folder = './data/'

	data = io.loadmat('./data/ex1.mat')
	X_3D = data['X_3D'][0]
	TVecs = data['TVecs']		# Translation vector: as the world origin is seen from the camera coordinates
	RMats = data['RMats']		# Rotation matrices: converts coordinates from world to camera
	kc = data['dist_params']	# Distortion parameters
	Kintr = data['intinsic_matrix']	# K matrix of the cameras
	
	imgs = [cv.imread(base_folder+str(i).zfill(5)+'.jpg') for i in range(TVecs.shape[0])]

	for imageIndex in range(0,TVecs.shape[0]):
		project_and_draw(imgs[imageIndex], X_3D, Kintr, RMats[imageIndex], TVecs[imageIndex], str(imageIndex).zfill(5)+'.jpg' ,False, kc)