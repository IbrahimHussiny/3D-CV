import cv2 as cv
import numpy as np
import scipy.io as io
import os
import sys
import numpy.linalg as linalg
# Global varialbles

# Downscaling factor
SCALE_FACTOR = 2
# Experiment with part of the image, in your submission try to send the result for the full resolution
H_BOUND = (10, 150)
W_BOUND = (10, 10+500)

EXPNAME = 'exp/'
os.system('mkdir ./' + EXPNAME)

calib = io.loadmat('./data/kitti/pose_and_K.mat')

K = calib['K']
POSE = calib['Pose']
BASELINE= calib['Baseline']
POSE[0:2,0:2] /= SCALE_FACTOR
F = POSE[0,0]

MIN_DISPARITY = int(4/SCALE_FACTOR)
MAX_DISPARITY = int(100/SCALE_FACTOR)
DOFFS = 0

# Outlier filter threshold for ssd
O_F_THRESHOLD = 2.0
SIMILARITY_MERTIC = 'ssd'
K_SIZE = 5

# The following three variables will be used by the ply creation function,
pre_text1 = """ply
format ascii 1.0"""
pre_text2 = "element vertex"
pre_text3 = """property float x
property float y
property float z
end_header"""
def ply_creator(input_, filename):
	assert (input_.ndim==2),"Pass 3d points as NumPointsX3 array "
	pre_text22 = pre_text2 + " "+str(input_.shape[0])
	pre_text11 = pre_text1
	pre_text33 = pre_text3
	fid = open(filename + '.ply', 'w')
	fid.write(pre_text11)
	fid.write('\n')
	fid.write(pre_text22)
	fid.write('\n')
	fid.write(pre_text33)
	fid.write('\n')
	for i in range(input_.shape[0]):
		# Check if the depth is not set to zero
		if input_[i,2]!=0:
			for c in range(2):
				fid.write(str(input_[i,c]) + ' ')
			fid.write(str(input_[i,2]))
			if i!=input_.shape[0]-1:
				fid.write('\n')
	fid.close()

"""
sub_pixel_crop() and pixel_interpolate(): are provided incase you want to do sub pixel matching.
sub_pixel_crop() will allow you to crop a patch centered at a floating point location like (10.89, 3.88).
You DO NOT HAVE to change them.
"""
def pixel_interpolate(img, h_idx, w_idx):
	"""
	- img: input image
	- h_idx, w_idx: center of the patch ( in the vertical and horizontal dimensions, respectively)
				  h_idx & w_idx could be integer of floating values.
	- pixel_out: output pixel after applying bilinear interpolation
	"""
	pixel_out = np.zeros((3,))
	h_s = np.int(h_idx)
	w_s = np.int(w_idx)
	for h in range(h_s, h_s+2):
		for w in range(w_s, w_s+2):
			weight = (1-np.abs(h_idx-h))*(1-np.abs(w_idx-w))
			pixel_out += weight*img[h,w,:]
	return pixel_out

def sub_pixel_crop(img, h_center, w_center, k_size):
	"""
	- img: input image
	- h_center, w_center: center of the patch ( in the vertical and horizontal dimensions, respectively)
				  h_center & w_center could be integer of floating values.
	- k_size: kernel width/ and height/
	- crop_out: output patch of size kxk
	"""
	crop_out = np.zeros((k_size, k_size, 3))
	offset = np.floor(k_size/2)
	h_idxs = np.linspace(h_center-offset, h_center+offset, k_size)
	w_idxs = np.linspace(w_center-offset, w_center+offset, k_size)
	for h in range(len(h_idxs)):
		for w in range(len(w_idxs)):
			crop_out[h,w,:] = pixel_interpolate(img, h_idxs[h], w_idxs[w])
	return crop_out

def copy_make_border(img, k_size):
	"""
	Patches centered at the edge of the image need additional padding of size k_size/2
	This function applies cv.copyMakeBorder to extend the image by k_size/2 in top, bottom, left and right part of the image
	"""
	offset = np.int(k_size/2.0)
	return cv.copyMakeBorder(img, top=offset, bottom=offset, left=offset, right=offset, borderType=cv.BORDER_REFLECT)

def write_depth_to_image(depth, f_name):
    """
    This function writes depth map to f_name
    """
    assert (depth.ndim==2),"Depth map should be a 2D array "
    max_depth = np.max(depth)
    min_depth = np.min(depth)
    depth_v1 = 255 - 255*((depth-min_depth)/max_depth)
    #depth_v2 = 255*((depth-min_depth)/max_depth)
    cv.imwrite(f_name, depth_v1)
    return True

def depth_to_3d(Z, x,y ):
    """
    Given the pixel position(y,x) and depth(Z)
    It computes the 3d point in world coordinates,
    first back-project the point from homogeneous image space to 3D,  by multiplying it with inverse of the camera intrinsic matrix,  inv(K)
    Then scale it so that you will get the point at depth equal to Z.
    the scale the vector
    """
    X_W = Z*np.dot(np.linalg.inv(K),np.asarray([[x],[y],[1]]))
    X_W = X_W.reshape(3,1)
    # returns a list [X, Y, Z]
    return [X_W[0,0], X_W[1,0], X_W[2,0]]

def is_outlier(score_vector):
    """
    O_F_THRESHOLD is the Outlier Filtering Threshold.
    For 'ncc' metric: if more than O_F_THRESHOLD number of disparity values have ncc score greater of equal to 0.8*max_score, the function returns True.
    For 'ssd' metric: if more than O_F_THRESHOLD number of disparity values have ssd score less than of equal to 1.4*min_score, the function returns True.
    The above thresholds are chosen by the TAs. Feel free, to try out different values.
    - The input to this function should be a 1D array of ncc scores computed for all disparities.
    """
    assert (score_vector.ndim==1), "Pass 1D array of vector of scores "
    if SIMILARITY_MERTIC=='ncc':
		max_score = np.max(score_vector)
		if np.sum(score_vector>=0.8*max_score)>=O_F_THRESHOLD:
			return True
		else:
			return False
    if SIMILARITY_MERTIC=='ssd':
		min_distance = np.min(score_vector)
		if np.sum(score_vector<=1.4*min_score)>=O_F_THRESHOLD:
			return True
		else:
			return False
    return True

def disparity_to_depth(disparity):
    """
    Converts disparity to depth.
    """
    inv_depth = (disparity+DOFFS)/(BASELINE*F)
    return 1/inv_depth

def ncc(patch_1, patch_2):
    """
    TODO
    1. Normalise the input patch by subtracting its mean and dividing by its standard deviation.
    Perform the normalisation for each color channel of the input patch separately.
    2. Reshape each normalised image patch into a 1D feature vector and then
    compute the dot product between the resulting normalised feature vectors.
    """
    raise NotImplementedError

def ssd(feature_1, feature_2):
    """
    TODO
    Compute the sum of square difference between the input features
    """
    raise NotImplementedError

def stereo_matching(img_left, img_right, K_SIZE, disp_per_pixel):
	"""
    This is on of the functions that you have to write.
    For the region of the leftimage delimited by H_BOUND and H_BOUND compute the following
        1. Compute Depth-Map(WITH and WITHOUT outlier filtering) and save them it as results/f_depth.png and results/un_f_depth.png.
        2. Create filtered and unfiltered point cloud by giving depth map to depth_to_3d(Z, y, x) function.
        3. Using ply_creator(), save point cloud:
           a. as results/un_f_pcl.ply for the depth map created WITHOUT outlier filtering.
           b. for the depth-map created WITH outlier filtering, as results/f_pcl.ply(), and

    While computing depth, put depth for outliers as 0. This way the ply_creator() fucntion will detect the
    ignore them without computing 3D points for them.

    Note that: the left and right_images are padded; by int(K_SIZE/2) on the left, right, top and bottom,
    """
    # TASKs:
    # Iterate over every pixel of the left image (with in the region bounded by H_BOUND and W_BOUND), (lets denote a sample as P_x_y)
        # For every P_x_y:
         # compute disparity using ssd cost metric
         # Convert disparity to depth, use disparity_to_depth()
         # Convert depth to point cloud, use depth_to_3d()
    # Save point clouds as .ply and depth as .png files
	disparties= np.zeros((H_BOUND[1] - H_BOUND[0],W_BOUND[1]- W_BOUND[0]))
	depths = np.zeros((H_BOUND[1]- H_BOUND[0],W_BOUND[1]-W_BOUND[0]))
	points3D = []
	for i in range (H_BOUND[0],H_BOUND[1]):
		for j in range(W_BOUND[0],W_BOUND[1]):
			left_pixel = img_left[i][j]
			match_diff = sys.maxint
			disparity = sys.maxint
			for z in range(j-MAX_DISPARITY,j-MIN_DISPARITY):
				right_pixel = img_right[i][z]
				pixel_diff = abs(left_pixel-right_pixel)
				#print pixel_diff
				if pixel_diff< match_diff:
					match_diff = pixel_diff
					disparity = abs(z-j)
			
			disparties[i-H_BOUND[0]][j-W_BOUND[0]]  =disparity
			pixel_depth = disparity_to_depth(disparity)
			depths[i-H_BOUND[0]][j-W_BOUND[0]] = pixel_depth
			points3D.append(depth_to_3d(pixel_depth,j-W_BOUND[0],i-H_BOUND[0]))

	cv.imwrite(EXPNAME+ "KITTI1_disparity_7.png",disparties)		
	write_depth_to_image(depths, EXPNAME+ "KITTI1_depth_7.png")
	ply_creator(np.array(points3D),EXPNAME+'KITTI1_reconstruction')
def main():
	# Set parameters
	# Use 1: for pixel-wise disparity
	# Optional: you can try to use sub-pixel level disparity, using, disp_per_pixel = n, n>1
    # Do not try this until you see the result for n=1
	disp_per_pixel = 1
	# Read images and expand borders
	# load Left image
	l_file = './data/kitti/l.png'
	l_im = cv.imread(l_file,0)
	resized_l_img = cv.pyrDown(l_im, SCALE_FACTOR)
	left_img = copy_make_border(resized_l_img, K_SIZE)
	# load Right image
	r_file = './data/kitti/r.png'
	r_im = cv.imread(r_file,0)
	resized_r_img = cv.pyrDown(r_im, SCALE_FACTOR)
	right_img = copy_make_border(resized_r_img, K_SIZE)
	# TODO: #1 fill in the stereo_matching() function called below
	dummy = stereo_matching(left_img, right_img, K_SIZE, disp_per_pixel)
main()
