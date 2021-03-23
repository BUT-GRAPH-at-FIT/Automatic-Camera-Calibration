.. _interface_label:

Interface description
=============================================================

The only necessary package is *calibration* which contains class *Calibration*. This class contains single method:

.. method:: calibrate(objects[, dims=(1920,1080), method='landmarks', calibration_bounds=[(1000, 10000), (90, 135), (-20, 20), (-15, 15), (5, 100)], iters=200])

    Parameters:

    * **objects** -- Detected keypoints in specific format as is described in :ref:`input_label`
    * **dims** -- Image dimensions in (width, height) format, serves for setting of the principal point [*tuple*, *list*]
    * **method** -- Which calibration method should be used (see :ref:`information_label`) ['Landmarks', 'OptInOpt', 'Plane']
    * **calibration_bounds** -- Optimization bounds where calibration parameters will be searched [*list of tuples*]

        * [(focal length lower, focal length upper), (rx lower, rx upper), (ry lower, ry upper), (rz lower, rz upper), (tz lower, tz upper)] -- order must be kept, for 'Plane' method only focal length bounds are used
        * [(focal length lower, focal length upper)] -- only focal length bounds can be set, the others stay as default (suitable mainly for 'Plane' method)

    * **iters** -- Iterations of calibration optimization (only used for 'OptInOpt' method) [*int*]
