from common import *
from scipy.optimize import differential_evolution
from itertools import combinations


class LandmarkCalib:
    '''
        Class for landmark calibration algorithm
    '''

    def __init__(self, bounds):
        self.calibration_bounds = bounds


    def calibrate(self, objects, dims):
        '''
        Whole calibration process
        '''
        self.pp = np.array([dims[0]/2, dims[1]/2])
        for o in objects:
            o['error'] = 1.0
            o['2d'] = np.array(o['2d'])
            o['3d'] = np.array(o['3d'])

        self.min_cnt = np.amin([len(o['2d']) for o in objects])
        self.max_cnt = np.amax([len(o['2d']) for o in objects])

        calib1 = self.landmark_calib(objects, bounds=self.calibration_bounds)

        get_weights(objects, self.focal, self.pp)

        calib = self.landmark_calib(objects, bounds=self.calibration_bounds)

        return calib


    def dist_indices(self, objects):
        '''
        Get indices for link each possible couple of deteced landmarks for each car
        '''
        combs = {}
        for i in range(self.min_cnt, self.max_cnt+1):
            combs[i] = np.array(list(combinations(range(i), 2)))[:,:,np.newaxis]

        combCnt = 0
        for o in objects:
            combCnt += len(combs[len(o['2d'])])

        indices = np.empty((combCnt,2,1), dtype=np.int32)
        weights = np.empty((combCnt,1), dtype=np.float64)

        cntComb, cntPts = 0, 0
        for o in objects:
            lPts = len(o['2d'])
            lComb = len(combs[lPts])
            indices[cntComb:cntComb+lComb,:,:] = combs[lPts]+cntPts
            weights[cntComb:cntComb+lComb,:] = np.power((1/o['error']), 4.0)
            cntComb += lComb
            cntPts += lPts

        return indices, weights, cntPts


    def compute_loss(self, optParams, fixedParams):
        '''
        Loss function for optimization
        '''
        P2D, Z, W, distsReal, indices, cntPts = fixedParams
        f, rx, ry, rz, tz = optParams

        K = construct_intrinsic_matrix_TF(tf.constant(f, dtype=tf.float64), tf.constant(self.pp, dtype=tf.float64))
        R = construct_rotation_matrix_TF(rx,ry,rz)
        T = tf.reshape(tf.pad(tf.reshape(tz,[1]), [[2,0]]), (3,1))

        K = tf.tile(tf.expand_dims(K, 0), [cntPts,1,1])
        R = tf.tile(tf.expand_dims(R, 0), [cntPts,1,1])
        T = tf.tile(tf.expand_dims(T, 0), [cntPts,1,1])

        P3DPROJ = pos_3d_from_2d_projection_TF(P2D, K, R, T, Z)

        couples = tf.gather_nd(P3DPROJ, indices)
        distsProj = tf.norm(couples[:,0,:]-couples[:,1,:], axis=1)

        loss = tf.reduce_sum(tf.square(((distsProj-distsReal)/distsReal))*W)/tf.reduce_sum(W)

        return loss.numpy()


    def landmark_calib(self, objects, bounds=[(1000, 10000), (90, 135), (-20, 20), (-20, 20), (10, 100)]):
        '''
        LadmarksCalib algorithm
        '''
        ind, weights, cntPts = self.dist_indices(objects)
        indices = tf.constant(ind)
        W = tf.constant(weights)

        pts2d = np.empty((cntPts, 3))
        pts3d = np.empty((cntPts, 4))
        cnt = 0
        for o in objects:
            l = len(o['2d'])
            pts2d[cnt:cnt+l,:] = np.insert(o['2d'], 2, 1, axis=1)
            pts3d[cnt:cnt+l,:] = np.insert(o['3d'], 3, 1, axis=1)
            cnt += l

        z = np.array(pts3d[:,2])[:,np.newaxis]

        P2D = tf.constant(pts2d)
        P3D = tf.constant(pts3d)
        Z = tf.constant(z)
        distsReal = tf.gather_nd(P3D, indices)
        distsReal = tf.expand_dims(tf.norm(distsReal[:,0,:]-distsReal[:,1,:], axis=1), -1)


        fixedParams = (P2D, Z, W, distsReal, indices, cntPts)

        result = differential_evolution(self.compute_loss, bounds, args=(fixedParams,), popsize=15, maxiter=100, recombination=0.9, disp=False)

        f,rx,ry,rz,tz = result.x

        self.focal = f
        K = construct_intrinsic_matrix(f, self.pp)
        R = construct_rotation_matrix(rx, ry, rz)
        T = np.array([[0],[0],[tz]])
        return {'K' : K, 'R' : R, 'T' : T, 'P' : projection_matrix(K, R, T)}
        # return {'f' : f, 'rx' : rx, 'ry' : ry, 'rz' : rz, 'tz' : tz, 'K' : K, 'R' : R, 'T' : T, 'P' : projectionMatrix(K, R, T)}
