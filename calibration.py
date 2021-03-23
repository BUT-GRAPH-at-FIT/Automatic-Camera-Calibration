import tensorflow as tf
if int(tf.__version__[0]) == 1:
    tf.compat.v1.enable_eager_execution()
from landmarks_calib import LandmarkCalib
from plane_calib import PlaneCalib
from opt_in_opt import OptInOpt

'''
Package with all LandmarkCalib implementation
Input - [{'2d' : [[x,y], [x,y], [x,y], ...], '3d' : [[x,y,z], [x,y,z], [x,y,z], ...]}, {'2d' : [[x,y], [x,y], [x,y], ...], '3d' : [[x,y,z], [x,y,z], [x,y,z], ...]}, ...]
'''

class Calibration:

    def __init__(self):
        pass


    def calibrate(self,
                 objects,
                 dims=(1920, 1080),
                 method='Landmarks',
                 calibration_bounds=[(1000, 10000), (90, 135), (-20, 20), (-15, 15), (5, 100)],
                 iters=200
                 ):

        method = method.lower()

        if method == 'landmarks':
            if len(calibration_bounds) == 1:
                calibration_bounds = [(calibration_bounds[0][0], calibration_bounds[0][1]), (90, 135), (-20, 20), (-15, 15), (5, 100)]
            for b in calibration_bounds:
                if type(b) != tuple or len(b) != 2:
                    sys.stderr.write('Bounds are not set correctly!')
                    exit()
            cal = LandmarkCalib(calibration_bounds)
        elif method == 'plane':
            if len(calibration_bounds) != 1:
                calibration_bounds =  [(calibration_bounds[0][0], calibration_bounds[0][1])]
            cal = PlaneCalib(calibration_bounds)
        elif method == 'optinopt':
            if len(calibration_bounds) == 1:
                calibration_bounds = [(calibration_bounds[0][0], calibration_bounds[0][1]), (90, 135), (-20, 20), (-15, 15), (5, 100)]
            for b in calibration_bounds:
                if type(b) != tuple or len(b) != 2:
                    sys.stderr.write('Bounds are not set correctly!')
                    exit()
            cal = OptInOpt(calibration_bounds, iters)

        return cal.calibrate(objects, dims)
