from common import *
from cv2 import solvePnP, SOLVEPNP_EPNP, Rodrigues
from scipy.optimize import minimize_scalar
import numpy as np


def rotation_from_vectors(vec1, vec2):
    '''
        Returns rotation matrix for transformation (rotation) between two vectors - Rodrigues formula
        https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    '''
    #vector (axis) of rotation
    u = np.cross(vec1, vec2) / np.linalg.norm(np.cross(vec1, vec2))
    #angle of rotation around axis u
    theta = np.arccos(np.dot(vec1, vec2)/(np.dot(np.linalg.norm(vec1), np.linalg.norm(vec2))))

    return Rodrigues(u*theta)[0]


def get_plane_weighted(pts, weights):
    '''
        Fit plane to the set of 3d points with set weights
    '''

    A = pts[:,:2]
    A = np.matrix(np.insert(A, A.shape[1], 1, axis=1))
    B = np.matrix(pts[:,2]).T
    W = np.matrix(np.diag(weights))

    fit = (A.T * W * A).I * A.T * W * B

    return np.squeeze(np.asarray(fit))


def get_plane_normal(plane):
    '''
        Get normal vector of the plane
    '''
    a, b, c = plane

    #get normal vector of the plane - three points on the plane (potentialy random)
    actPts = [[0,0,0], [0,5,0], [5,0,0]]
    for p in actPts:
        p[2] = a * p[0] + b * p[1] + c
        p = list(p)

    actPts = np.array(actPts)
    #vectors between points
    PQ = actPts[1] - actPts[0]
    PR = actPts[2] - actPts[0]
    #normal vector of the plane as cross product
    norm = np.cross(PQ, PR)
    #normalize vector - make unit
    return norm / np.linalg.norm(norm)


class PlaneCalib:
    '''
        Class for plane calibration algorithm
    '''

    def __init__(self, bounds):
        self.calibration_bounds = bounds
        self.val = 300


    def calibrate(self, objects, dims):

        self.pp = np.array([dims[0]/2, dims[1]/2])

        for o in objects:
            o['2d'] = np.array(o['2d'])
            o['3d'] = np.array(o['3d'])
        get_combinations(objects)

        f = self.get_focal(objects)
        self.get_weights(objects, f)

        K = construct_intrinsic_matrix(f, self.pp)

        return self.plane_calib(objects, K)


    def get_weights(self, objects, f):
        '''
        Compute objects' weights - PnP reprojection error with known focal
        '''
        K = construct_intrinsic_matrix(f, self.pp)

        for o in objects:
            retVal, rVec, tVec = solvePnP(o['3d'], np.reshape(o['2d'], (o['2d'].shape[0],1,2)), K, None, flags=SOLVEPNP_EPNP)
            R = Rodrigues(rVec)[0]
            P = projection_matrix(K, R, tVec)
            pts2dProj = project_3d_to_2d(np.insert(o['3d'], 3, 1, axis=1).T, P)[:,:2]
            o['error'] = fit_error_RSE(o['2d'], pts2dProj)


    def plane_calib(self, objects, K):

        all_pts = []
        #pts = np.array([[0,0,0,1], [5,5,0,1], [-5,-5,0,1], [5,-5,0,1], [-5,5,0,1]])
        pts = np.array([[0,0,0,1]])
        for o in objects:

            #PnP for each car - position of the object due to camera position
            retVal, rVec, tVec = solvePnP(o['3d'], np.reshape(o['2d'], (o['2d'].shape[0],1,2)).astype(np.float32), K, None, flags=SOLVEPNP_EPNP)

            R = Rodrigues(rVec)[0]

            #projection matrix of the object
            modelP = np.concatenate((R,tVec), axis=1)
            modelP = np.vstack([modelP, [0,0,0,1]])


            ref_pt = np.dot(modelP, pts.T).T

            all_pts += list(ref_pt)

        all_pts = np.array(all_pts)


        act_weights = np.array([(1/o['error']) for o in objects])

        fit_calib = get_plane_weighted(all_pts, act_weights)
        norm = get_plane_normal(fit_calib)

        cam = np.array([0,0,1])
        #get rotation matrix between two vectors (camera, normal of the plane)
        R = rotation_from_vectors(cam, norm)
        tVec = np.array([[0],[0],[fit_calib[2]]])

        self.calib = {'K' : K, 'R' : R, 'T' : tVec, 'P' : projection_matrix(K, R, tVec)}

        return self.calib


    def focal_cost_function(self, opt_params, fixed_params):
        f = opt_params

        objects = fixed_params

        K = construct_intrinsic_matrix(f, self.pp)

        self.get_weights(objects, f)

        used_objects = sorted(objects, key=lambda o:o['error'])[:self.val]

        calib = self.plane_calib(used_objects, K)
        R = calib['R']
        T = calib['T']

        errors = []
        weights = []
        #for each targets compute RRSE (root relative squared error) between projected and real keypoints
        for obj in used_objects:

            obj['3d_computed'] = []

            for pos in range(len(obj['3d'])):
                obj['3d_computed'].append(np.squeeze(pos_3d_from_2d_projection(obj['2d'][pos], K, R, T, obj['3d'][pos][2])))


            #each keypoints combination
            for c in obj['combinations']:
                #compute 3d position of both outer keypoints
                p1 = obj['3d_computed'][c['comb'][0]]
                p2 = obj['3d_computed'][c['comb'][1]]

                eAct = np.abs(c['dist'] - np.linalg.norm(p1-p2)) / c['dist']

                errors.append(eAct)
                weights.append(1/obj['error'])

        # mean of all objects' errors
        return np.average(errors, weights=weights)
        # return np.sqrt(np.mean(errors))


    def get_focal(self, objects):
        result = minimize_scalar(self.focal_cost_function, bounds=self.calibration_bounds[0], method='Bounded', args=(objects,))

        f = result.x

        return f
