from de import my_differential_evolution, my_differential_evolution_single
from common import *
from cv2 import solvePnP, SOLVEPNP_EPNP, Rodrigues
import numpy as np


def get_errors(data, pp):
    '''
    Get reprojection errors of all cars
    '''
    for d in data:
        d['error'] = []
        for f in np.random.uniform(1000, 10001, 10):
            K = construct_intrinsic_matrix(f, pp)
            retVal, rVec, tVec = solvePnP(d['3d'], np.reshape(d['2d'], (d['2d'].shape[0],1,2)), K, None, flags=SOLVEPNP_EPNP)
            R = Rodrigues(rVec)[0]
            P = projection_matrix(K, R, tVec)
            pts2dProj = project_3d_to_2d(np.insert(d['3d'], 3, 1, axis=1).T, P)[:,:2]
            d['error'].append(reprojection_error(d['2d'], pts2dProj))
        d['error'] = np.mean(d['error'])


def get_cars_error(oio, P):
    errors = []
    for pos in range(len(oio.pts2d_pop)):
        trans = my_differential_evolution(get_car_transformations, oio.bounds_loc, [oio.pts2d_pop[pos], oio.pts3d_pop[pos], P], iter=oio.iters_loc, max_same_iter=oio.iters_same_loc, popsize=oio.popsize_loc, mutation=(0.5, 1.0), recombination=0.7, disp=False, batch_size=oio.pts2d_pop[pos].shape[0])
        trans_mat = construct_inplane_transformation_matrix_single(trans)
        trans_pts = tf.transpose(tf.matmul(trans_mat, tf.transpose(oio.pts3d[pos], perm=[0,2,1])), perm=[0,2,1])
        pts_proj = project_3d_to_2d_TF_single(trans_pts, expand_to_batch(P, oio.pts2d_pop[pos].shape[0]))
        errors.append(reprojection_error_TF_single(oio.pts2d[pos], pts_proj))

    errors = tf.concat(errors, 0)

    return tf.reduce_sum(errors * oio.weights, axis=0) / tf.reduce_sum(oio.weights, axis=0)


def get_car_transformations(opt_params, fixed_params):
    pts2d, pts3d, P = fixed_params
    trans_mat = construct_inplane_transformation_matrix(opt_params)
    trans_pts = transform_3d_pts(pts3d, trans_mat)
    trans_proj = project_3d_to_2d_TF(trans_pts, expand_to_batch_and_population_2d(P, pts2d.shape[0], pts2d.shape[1]))
    return reprojection_error_TF(pts2d, trans_proj)


def compute_calibration_error(opt_params, fixed_params):
    oio = fixed_params
    f, rx, ry, rz, tz = tf.transpose(opt_params)
    errors = []
    for pos in range(oio.popsize_cal):
        K = construct_intrinsic_matrix_TF(f[pos], tf.constant(oio.pp, dtype=tf.float64))
        R = construct_rotation_matrix_TF(rx[pos],ry[pos],rz[pos])
        T = construct_translation_vector_TF(tf.constant(0, dtype=tf.float64),tf.constant(0, dtype=tf.float64),tz[pos])
        P = projection_matrix_TF(K, R, T)
        errors.append(get_cars_error(oio, P))

    return tf.convert_to_tensor(errors)


class OptInOpt:
    '''
        Class for OptInOpt calibration algorithm
    '''

    def __init__(self, bounds, iters):

        self.popsize_loc = 25
        self.popsize_cal = 20
        self.iters_loc = 200
        self.iters_same_loc = 20
        self.iters_cal = iters
        self.iters_same_cal = 50
        self.bounds_loc = [(-150,150), (-150,150), (0,360)]
        self.bounds_cal = bounds

        while self.popsize_loc%3 != 0:
            self.popsize_loc += 1
        while self.popsize_cal%3 != 0:
            self.popsize_cal += 1


    def set_data(self, data):

        min_cnt, max_cnt = 1000, 0
        for d in data:
            if len(d['2d']) > max_cnt:
                max_cnt = len(d['2d'])
            if len(d['2d']) < min_cnt:
                min_cnt = len(d['2d'])

        pts2d, pts3d = {}, {}
        errors = []
        for i in range(min_cnt, max_cnt+1):
            pts2d[i] = []
            pts3d[i] = []

        for d in data:
            pts2d[len(d['2d'])].append(np.insert(d['2d'], 2, 1, axis=1))
            pts3d[len(d['2d'])].append(np.insert(d['3d'], 3, 1, axis=1))
            errors.append(d['error'])

        self.pts2d = [np.array(pts2d[k], dtype=np.float64) for k in pts2d if len(pts2d[k])]
        self.pts3d = [np.array(pts3d[k], dtype=np.float64) for k in pts3d if len(pts2d[k])]
        self.weights = tf.convert_to_tensor(np.reciprocal(errors), dtype=tf.float64)

        self.pts2d_pop = [tf.tile(tf.expand_dims(tf.convert_to_tensor(p2), 1), [1,self.popsize_loc,1,1]) for p2 in self.pts2d]
        self.pts3d_pop = [tf.tile(tf.expand_dims(tf.convert_to_tensor(p3), 1), [1,self.popsize_loc,1,1]) for p3 in self.pts3d]


    def calibrate(self, objects, dims):
        self.pp = np.array([dims[0]/2, dims[1]/2])
        for o in objects:
            o['2d'] = np.array(o['2d'])
            o['3d'] = np.array(o['3d'])

        get_errors(objects, self.pp)
        self.set_data(objects)

        cal = my_differential_evolution_single(compute_calibration_error, self.bounds_cal, self, iter=self.iters_cal, max_same_iter=self.iters_same_cal, popsize=self.popsize_cal, mutation=(0.5, 1.0), recombination=0.7, disp=True)
        K = construct_intrinsic_matrix(cal[0], self.pp)
        R = construct_rotation_matrix(cal[1], cal[2], cal[3])
        T = construct_translation_vector(0,0,cal[4])
        calib = {'K' : K, 'R' : R, 'T' : T, 'P' : projection_matrix(K, R, T)}
        return calib
