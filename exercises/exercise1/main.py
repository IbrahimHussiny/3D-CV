import os, sys
import numpy as np
from numpy.linalg import inv
import cv2 as cv
import cmath

import scipy.io as io

outputFolderPath = "./outputs/"

#create output folder if not exists
os.makedirs(outputFolderPath)

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

def correctRadialDistortion(pixelPoints, K, dist_params):
	results = []
	for point in pixelPoints:
		#transform to normalized image coordinates
		normalizedPoint = np.dot(inv(np.array(K.reshape(3,3))) , np.array([point[0] , point[1] , 1]))
		# remove last 1 element
		normalizedPoint = np.delete(normalizedPoint, 2)

		r = cmath.sqrt(pow(normalizedPoint[0],2) + pow(normalizedPoint[1],2) )
		r= r.real
		distortionNumber = 1 + dist_params[0] * pow(r,2) + dist_params[1] * pow(r,4) + dist_params[4]* pow(r,6)		
		
		undistortedPoint = [x * distortionNumber for x in normalizedPoint]
		#convert back to pixel coordinates
		undistortedPoint = np.dot(np.array(K.reshape(3,3)), np.array([undistortedPoint[0],undistortedPoint[1],  1]))
		#remove last 1 homogenous element, and add to output list
		results.append([undistortedPoint[0], undistortedPoint[1]])
	
	return results

# Project points from 3d world coordinates to 2d image coordinates
def project_points(X, K, R, T):
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
		#recovering homogeneous coordinates by dividing over the last element, and add it to results
		cameraPoints.append(pixelPoint[:-1]/pixelPoint[-1])

	return cameraPoints

def project_and_draw(img, X_3d, K, R, T, image_name, distortion_flag, distortion_parameters):
	
	projectedPoints = project_points(X_3d, K, R,T)
	if distortion_flag == True:
		 undistortedPoints = correctRadialDistortion(projectedPoints, K, distortion_parameters)
		 for undistortedPoint in undistortedPoints:
			 cv.circle(img, tuple(undistortedPoint), 1, (0,255,0), -1)
	for projectedPoint in projectedPoints:
		cv.circle(img, tuple(projectedPoint), 1, (0,0,255),-1)
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
		project_and_draw(imgs[imageIndex], X_3D, Kintr, RMats[imageIndex], TVecs[imageIndex], str(imageIndex).zfill(5)+'.jpg' ,True, kc)