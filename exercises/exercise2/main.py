import numpy as np
from numpy.linalg import inv, det, svd
import scipy.io as io

REL_TOLERANCE = 0.0
ABS_TOLERANCE = 1e-05

data = io.loadmat('./data' + '/ex2.mat')

def isclose(a,b,rel_tol=REL_TOLERANCE,abs_tol=ABS_TOLERANCE):
    return abs(a-b) <= max(rel_tol*max(abs(a),abs(b)),abs_tol)

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def is_rotation_matrix(M):
    """M is rotation iff orthogonal matrix with determinant 1."""
    if not np.allclose(np.dot(M, np.transpose(M)), np.identity(3), REL_TOLERANCE, ABS_TOLERANCE):
        print "M dot M^T not close to I, but", np.dot(M, np.transpose(M))
        return False
    if not isclose(det(M), 1):
        print "Determinant != 1, but ", det(M)
        return False
    return True

def compute_relative_rotation(homography_name):
    # instrinsic paramters
    alpha_x = data['alpha_x'][0][0]
    alpha_y = data['alpha_y'][0][0]
    x_0 = data['x_0'][0][0]
    y_0 = data['y_0'][0][0]
    s = data['s'][0][0]

    # 3.1) load homography
    H = data[homography_name]

    # 3.2) compute R_rel (pure rotation)
    K = np.matrix([
        [alpha_x,   s,          x_0],
        [0,         alpha_y,    y_0],
        [0,         0,          1]])
    # since we assume no translation (t = [0 0 0]) we have H = KR(K^-1) -> R = (K^-1)HK
    R_rel = np.dot(np.dot(inv(K), H), K)

    # 3.3+3.4) check whether R_rel fulfills properties of Rotation-Matrix and if necessary, correct R_rel; then print
    if not is_rotation_matrix(R_rel):   # Do SVD to correct R2, and set singular matrix to I
        print 'Matrix found not to be a proper rotation matrix: ', R_rel
        U, W, V = svd(R_rel)
        R_rel = np.dot(U,V)
    print "Proper Rotationmatrix:", R_rel, "END"

def compute_pose(homography_name):
    # instrinsic paramters
    alpha_x = data['alpha_x'][0][0]
    alpha_y = data['alpha_y'][0][0]
    x_0 = data['x_0'][0][0]
    y_0 = data['y_0'][0][0]
    s = data['s'][0][0]

    H = data[homography_name]
    K = np.matrix([
        [alpha_x, s, x_0],
        [0, alpha_y, y_0],
        [0, 0, 1]])

    # since Z = 0 we have H = K[r1r2t]
    Rt = np.around(np.dot(inv(K),H))
    r1= normalize(np.array(Rt[:,0]))
    r2 = normalize(np.array(Rt[:,1]))
    r3 = np.cross(np.transpose(r1),np.transpose(r2))
    R = np.column_stack((r1,r2,np.transpose(r3)))
    print "R: ", R
    print "t: ", Rt[:,2]


if __name__ == '__main__':
    # Task 3
    print 'Task 3:\n'
    compute_relative_rotation('H1')
    compute_relative_rotation('H2')

    # Task 4
    print '\n\nTask4:\n'
    compute_pose('H3')