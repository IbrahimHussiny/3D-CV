import numpy as np
import numpy.linalg
import scipy.io as io
import cv2
import math
import random 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getCrossMultplicationMat(vec):
    return np.matrix([[0, -vec[2], vec[1]],[vec[2] , 0 , -vec[0]],[-vec[1], vec[0],0]])

def computFundamentalMat(K_0 , K_1, t_1, R_1):
    leftPart = np.dot(np.linalg.inv(K_1).T, getCrossMultplicationMat(t_1))
    rightPart = np.dot(R_1,np.linalg.inv(K_0)) 
    return np.dot(leftPart,rightPart)

#min distance between point in line in space, is the line perpendicular from point to the line
def getMinDistance(point, line):    
    numerator = math.fabs(line[0]*point[0] +line[1] * point[1] +line[2])
    denominator = math.sqrt((line[0]*line[0]) + (line[1]*line[1]))
    return numerator/denominator

def getBordaerInidcies(line):
    #x = 0 
    begin_index = (0,int((-line[2])/line[1]))
    #x =4752
    end_index = (4752,int((-line[2]-line[0]*4752)/line[1]))
    return begin_index,end_index

def drawEpipolarLines(img, lines):
    new_img = img.copy()
    for epipolar_line in lines:
        begin_index , end_index = getBordaerInidcies(epipolar_line)    
        cv2.line(new_img,begin_index,end_index,(255,0,0),3)
    cv2.imwrite("epilines.jpg",new_img)

def draw_matched_features(img, y_displacement, features_0, features_1, line_feature_dict):
    for i in range(0,len(features_1)):
        matched_feature = features_1[line_feature_dict[i]]
        
        #matched feature1 will have displacment in y direction
        matched_feature = [int(matched_feature[0]) , int(matched_feature[1]+y_displacement)]
        original_feature = [int(features_0[i][0]), int(features_0[i][1])]
        cv2.line(img, tuple(original_feature),tuple(matched_feature),[random.randint(0,255),random.randint(0,255),random.randint(0,255)],3)
    
    cv2.imwrite('matches.jpg', img)

def getExtrinsicParametersMat(RotationMatrix, t):
	# upper left 3*3 matrix 
    extrinsicParametersMat = np.array(RotationMatrix)
    extrinsicParametersMat.reshape(3,3)
    #-RC (upper right matrix of the extrinsic parameters mat)
    RC = np.matrix([[t[0]],[t[1]],[t[2]]])
    extrinsicParametersMat = np.concatenate((extrinsicParametersMat, RC), axis =1)
	
	# add last row of the matrix [0,0,0,1]
    extrinsicParametersMat = np.vstack([extrinsicParametersMat, np.array([0,0,0,1])])
	
    return extrinsicParametersMat

def getIntrinsicParametersMat(K):
	#Intrinsic matrix = K[I3 | 03]
	return np.dot(np.array(K.reshape(3,3)),np.append(np.identity(3),np.zeros((3,1)), axis =1))

def getPMatrix(K,R,t):
    return np.dot(getIntrinsicParametersMat(K), getExtrinsicParametersMat(R,t))

def getAMatrix(P_0,P_1, point_0 , point_1):
    firstCamera_0 = np.subtract(np.dot(point_0[0],P_0[2,:]), P_0[0,:])
    firstCamera_1 = np.subtract(np.dot(point_0[1],P_0[0,:]), P_0[0,:])
    firstCamera_A = np.concatenate((firstCamera_0,firstCamera_1),axis = 0)

    secondCamera_0 = np.subtract(np.dot(point_1[0],P_1[2,:]), P_1[0,:])
    secondCamera_1 = np.subtract(np.dot(point_1[1],P_1[0,:]), P_1[0,:])
    secondCamera_A = np.concatenate((secondCamera_0,secondCamera_1),axis = 0)

    return np.concatenate((firstCamera_A,secondCamera_A),axis = 0)

def plot3DPoints(points_list):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for point in points_list:
        ax.scatter(point[0], point[1], point[2])
    fig.savefig('3D.png')    

#############################Main code##############################################################
data = io.loadmat('./data/' + 'data.mat')

K_0 = data['K_0']
K_1 =   data['K_1']
R_0 = np.identity(3)
R_1 =  data['R_1']
t_0 = np.zeros(3)
t_1 =   data['t_1'][0]
corners_0 =   data['cornersCam0']
corners_1 =   data['cornersCam1']

fundamentalMat =  computFundamentalMat(K_0,K_1, t_1, R_1)
print 'Fundamental Matrix = '
print fundamentalMat

#list of all epipolar_lines in image_1
epipolar_lines =[]
for feature in corners_0:
    epipolar_line = np.array(np.dot(fundamentalMat, np.array([feature[0], feature[1],1])))
    epipolar_lines.append(epipolar_line[0])

image_0 = cv2.imread("./data/Camera00.jpg")
image_1 = cv2.imread("./data/Camera01.jpg")
drawEpipolarLines(image_1,epipolar_lines)

#fill dict with matched feature for each epipolar line
epi_line_matched_feature = {}
for i  in range(0,len(epipolar_lines)):
    #any large number, should be INF
    min_distance = 4000000
    for j in range(0,len(corners_1)):
        dist = getMinDistance(corners_1[j], epipolar_lines[i])
        if(dist< min_distance):
            min_distance = dist
            epi_line_matched_feature[i] = j

#concatenate the two images to draw them in one img
mixed_image = np.concatenate((image_0,image_1), axis =0)
draw_matched_features(mixed_image,3168,corners_0,corners_1, epi_line_matched_feature)

######################################Part 2########################################################
#@TODO: Check if 3D points, and output figure are correct 

P_0 = getPMatrix(K_0,R_0,t_0)
P_1 = getPMatrix(K_1,R_1,t_1)

#list of the 3D points
points_list = []
for i in range(0,len(corners_0)):
    #get A matrix
    AMat = getAMatrix(P_0,P_1,corners_0[i],corners_1[i])
    #get eigenVectors of (A.T).A
    w,v = np.linalg.eig(np.dot(AMat.T,AMat))
    #get eigenvector with min eigevalue (solution for 3D point)
    minEigenVector = v[w.argmin(axis = 0)]
    minEigenVector = np.array(minEigenVector)
    minEigenVector.reshape(1,4)
    minEigenVector = minEigenVector[0]
    #divide over last elment to recover homogeneous coordinates
    points_list.append(np.divide(minEigenVector[:-1],minEigenVector[-1]))
#plot and save the 3D points
plot3DPoints(points_list)

