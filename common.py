import tensorflow as tf
import numpy as np
from math import pi as PI
from cv2 import solvePnP, SOLVEPNP_EPNP, Rodrigues
from itertools import combinations


def projection_matrix(K, R, T):
    '''
    Returns camera projection matrix P = K * [R,T]
    '''
    return np.dot(K,np.concatenate((R,T), axis=1))


def projection_matrix_TF(K, R, T):
    '''
    Returns camera projection matrix P = K * [R,T]
    '''
    return tf.matmul(K, tf.concat([R,T], 1))


def construct_rotation_matrix(rx, ry, rz):
    '''
    Construct rotation matrix from Euler angles (degrees) - NumPy implementation
    '''
    rx, ry, rz = np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])

    return np.dot(np.dot(Rz,Ry), Rx)


def construct_rotation_matrix_TF(rx, ry, rz):
    '''
    Construct rotation matrix from Euler angles (degrees) - Tensorflow implementation
    '''
    r_x = rx * PI / 180.0
    r_y = ry * PI / 180.0
    r_z = rz * PI / 180.0
    Rx = tf.reshape(tf.stack([1,0,0,0,tf.cos(r_x),-tf.sin(r_x),0,tf.sin(r_x),tf.cos(r_x)]), (3,3))
    Ry = tf.reshape(tf.stack([tf.cos(r_y),0,tf.sin(r_y),0,1,0,-tf.sin(r_y),0,tf.cos(r_y)]), (3,3))
    Rz = tf.reshape(tf.stack([tf.cos(r_z),-tf.sin(r_z),0,tf.sin(r_z),tf.cos(r_z),0,0,0,1]), (3,3))
    return tf.matmul(tf.matmul(Rz, Ry), Rx)


def construct_translation_vector(tx, ty, tz):
    return np.array([tx,ty,tz])[:,np.newaxis]


def construct_translation_vector_TF(tx, ty, tz):
    '''
    Construct translation vector with proper shape
    '''
    return tf.expand_dims(tf.stack([tx,ty,tz], 0), 1)


def construct_intrinsic_matrix(f, pp):
    '''
    Construct intrinsic matrix - NumPy implementataion
    '''
    return np.array([[f,0,pp[0]],[0,f,pp[1]],[0,0,1]], dtype=np.float64)


def construct_intrinsic_matrix_TF(f, pp):
    '''
    Construct intrinsic matrix - Tensorflow implementataion
    '''
    return tf.stack([tf.concat([tf.pad(tf.reshape(f,[1]), [[0,1]]), tf.reshape(pp[0],[1])], 0), tf.concat([tf.pad(tf.reshape(f,[1]), [[1,0]]), tf.reshape(pp[1],[1])], 0), tf.constant([0,0,1], dtype=tf.float64)])


def project_3d_to_2d(X, P):
    '''
    Projects points X to camera image defined projection matrix P
    '''
    x = np.dot(P,X)
    z = x[2, :]
    x = x / z
    return x.T


def project_3d_to_2d_TF(pts, P):
    '''
    pts <- (BATCH, POP, CNT, 4)
    P <- (BATCH, POP, 3, 3)
    out -> (BATCH, POP, CNT, 3)
    '''
    x = tf.matmul(P,tf.transpose(pts, perm=[0,1,3,2]))
    z = tf.expand_dims(x[:,:,2,:], 2)
    x = x/z
    return tf.transpose(x, perm=[0,1,3,2])


def project_3d_to_2d_TF_single(pts, P):
    '''
    pts <- (BATCH, CNT, 4)
    P <- (BATCH, 3, 3)
    out -> (BATCH, CNT, 3)
    '''
    x = tf.matmul(P, tf.transpose(pts, perm=[0,2,1]))
    z = tf.expand_dims(x[:,2,:], 1)
    x = x/z
    return tf.transpose(x, perm=[0,2,1])


def reprojection_error_TF(pts_source, pts_target):
    '''
    pts_source <- (BATCH, POP, CNT, 3)
    pts_target <- (BATCH, POP, CNT, 3)
    out -> (BATCH, POP)
    '''
    return tf.reduce_sum(tf.square(tf.norm(pts_target-pts_source, axis=3)), axis=2)

def reprojection_error_TF_single(pts_source, pts_target):
    '''
    pts_source <- (BATCH, CNT, 3)
    pts_target <- (BATCH, CNT, 3)
    out -> (BATCH)
    '''
    return tf.reduce_sum(tf.square(tf.norm(pts_target-pts_source, axis=2)), axis=1)


def reprojection_error(ptsSrc, ptsTar):
    return np.sum(np.square(np.linalg.norm(ptsSrc-ptsTar, axis=1)))


def pos_3d_from_2d_projection(pt, K, R, T, z):
    '''
        3D position from 2D projection with known calibration parameters and Z-coordinate - NumPy implementation
    '''
    pt2d = np.array((pt[0], pt[1], 1), dtype=np.float64).T
    kInv = np.linalg.inv(K)
    rInv = np.linalg.inv(R)

    rInv_kInv_pt = np.dot(np.dot(rInv, kInv), pt2d)
    rInv_t = np.dot(rInv, T)
    s = z + rInv_t[2]
    s /= rInv_kInv_pt[2]

    return np.dot(rInv, np.reshape(np.dot(s*kInv, pt2d), (3,1)) - T)


def pos_3d_from_2d_projection_TF(pt, K, R, T, Z):
    '''
    3D position from 2D projection with known calibration parameters and Z-coordinate
    pt - (N,3)
    K - (N,3,3)
    R - (N,3,3)
    T - (N,3,1)
    Z - (N,)
    returns:
        (N,3,1)
    '''
    pt2d = tf.expand_dims(tf.constant(pt), -1)
    kInv = tf.linalg.inv(K)
    rInv = tf.linalg.inv(R)

    rInv_kInv_pt = tf.matmul(tf.matmul(rInv, kInv), pt2d)
    rInv_t = tf.matmul(rInv, T)
    s = Z + rInv_t[...,2,:]
    s /= rInv_kInv_pt[...,2,:]
    s = tf.expand_dims(s, -1)

    return tf.matmul(rInv, tf.matmul(s*kInv, pt2d) - T)


def construct_inplane_transformation_matrix(trans_params):
    '''
    Construct matrices for transformation of 3D vehicle points in road plane - tx, ty translation and rz rotation
    trans_params <- (BATCH, POP, 3)
    out -> (BATCH, POP, 4, 4)
    '''
    tx = trans_params[:,:,0]
    ty = trans_params[:,:,1]
    rz = trans_params[:,:,2] * PI / 180.0
    angles = tf.pad(tf.concat([tf.expand_dims(tf.stack([tf.cos(rz), tf.sin(rz)], 2), -1), tf.expand_dims(tf.stack([-tf.sin(rz), tf.cos(rz)], 2), -1)], 3), [[0,0],[0,0],[0,0],[0,1]])
    trans = tf.expand_dims(tf.stack([tx, ty], 2), -1)
    top = tf.concat([angles, trans], 3)
    bottom = tf.pad(tf.eye(2, batch_shape=trans_params.shape[:2], dtype=tf.float64), [[0,0],[0,0],[0,0],[2,0]])

    return tf.concat([top,bottom], 2)

def construct_inplane_transformation_matrix_single(trans_params):
    '''
    Construct matrices for transformation of 3D vehicle points in road plane - tx, ty translation and rz rotation
    trans_params <- (BATCH, 3)
    out -> (BATCH, 4, 4)
    '''
    tx = trans_params[:,0]
    ty = trans_params[:,1]
    rz = trans_params[:,2] * PI / 180.0
    angles = tf.pad(tf.concat([tf.expand_dims(tf.stack([tf.cos(rz), tf.sin(rz)], 1), -1), tf.expand_dims(tf.stack([-tf.sin(rz), tf.cos(rz)], 1), -1)], 2), [[0,0],[0,0],[0,1]])
    trans = tf.expand_dims(tf.stack([tx, ty], 1), -1)
    top = tf.concat([angles, trans], 2)
    bottom = tf.pad(tf.eye(2, batch_shape=trans_params.shape[:1], dtype=tf.float64), [[0,0],[0,0],[2,0]])

    return tf.concat([top,bottom], 1)



def transform_3d_pts(pts_source, trans_mat):
    '''
    Transform 3D points by transformation matrix
    pts_source <- (BATCH, POP, CNT, 4)
    trans_mat <- (BATCH, POP, 4, 4)
    out -> (BATCH, POP, CNT, 4)
    '''
    return tf.transpose(tf.matmul(trans_mat, tf.transpose(pts_source, perm=[0,1,3,2])), perm=[0,1,3,2])


def fit_error_RSE(refPts, transPts):
    '''
    Realtive squared error of two points sets
    '''
    meanPoint = np.mean(refPts, axis=0)

    numerator, denominator = 0, 0

    for i in range(len(refPts)):
        numerator += (np.linalg.norm(refPts[i] - transPts[i]) * np.linalg.norm(refPts[i] - transPts[i]))
        denominator += (np.linalg.norm(refPts[i] - meanPoint) * np.linalg.norm(refPts[i] - meanPoint))

    if denominator == 0.0:
        return 100.0
    return np.sqrt(numerator/denominator)


def anglesFromRotationMatrix(R):
    '''
        Get Euler angles from rotation matrix (decompose) - in degrees
    '''
    sy = np.sqrt(R[0,0]*R[0,0]+R[1,0]*R[1,0])

    singular = sy < 1e-6

    if not singular:
        rx = np.arctan2(R[2,1], R[2,2])
        ry = np.arctan2(-R[2,0], sy)
        rz = np.arctan2(R[1,0], R[0,0])
    else:
        rx = np.arctan2(-R[1,2], R[2,2])
        ry = np.arctan2(-R[2,0], sy)
        rz = 0

    return [np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz)]


def get_combinations(objects):
    '''
    Compute distances between all possible pairs for each object
    '''
    for o in objects:
        #all possible paits
        comb = list(combinations(range(len(o['2d'])), 2))

        o['combinations'] = []
        for c in comb:
            dist = np.linalg.norm(o['3d'][c[0]]-o['3d'][c[1]])

            o['combinations'].append({'comb' : c, 'dist' : dist})


def get_weights(objects, focal, pp):
    '''
    Compute objects' weights - PnP reprojection error with known focal
    '''
    K = construct_intrinsic_matrix(focal, pp)

    for o in objects:
        retVal, rVec, tVec = solvePnP(o['3d'], np.reshape(o['2d'], (o['2d'].shape[0],1,2)), K, None, flags=SOLVEPNP_EPNP)
        R = Rodrigues(rVec)[0]
        P = projection_matrix(K, R, tVec)
        pts2dProj = project_3d_to_2d(np.insert(o['3d'], 3, 1, axis=1).T, P)[:,:2]
        o['error'] = fit_error_RSE(o['2d'], pts2dProj)


def expand_to_batch(x, batch_size):
    return tf.tile(tf.expand_dims(x, 0), [batch_size,1,1])


def expand_to_batch_and_population_2d(x, batch_size, population_size):
    '''
    Expand (tile) tensor to proper batch size and population size
    '''
    return tf.tile(tf.expand_dims(tf.expand_dims(x, 0), 0), [batch_size,population_size,1,1])
