.. _usage_label:

Usage
=============================================================

Here are the main information how to use the camera calibration tool.

.. _input_label:

Input
---------------------------

The input to the calibration tool are 2D detected keypoints within image plane (not a part of the tool) and corresponding 3D keypoints locations in object coordinate system (OCS).
This 3D OCS is unique for each detected object --- not a common coordinate system for all detections.
The input must be in specific format:

.. code-block:: python3

    [
        {'2d' : [[x,y], [x,y], [x,y], ...],
         '3d' : [[x,y,z], [x,y,z], [x,y,z], ...]},
        {'2d' : [[x,y], [x,y], [x,y], ...],
         '3d' : [[x,y,z], [x,y,z], [x,y,z], ...]},
         ...,
        {'2d' : [[x,y], [x,y], [x,y], ...],
         '3d' : [[x,y,z], [x,y,z], [x,y,z], ...]}
    ]

The input must be a list of dictionaries. Each dictionary contains two keys ('2d', '3d') and fit to single object's observation. Each key is composed of list of tuples with proper 2D or 3D keypoints positions.
Corresponding 2D-3D locations must have the same length. Count of keypoints can differ for each observation. This variable is passed to the proper function as is described in :ref:`interface_label`.

Output
---------------------------

The output of the function calling is as follows:

.. code-block:: python3

    {
     'K': numpy.ndarray(shape=(3, 3), dtype=float64),
     'R': numpy.ndarray(shape=(3, 3), dtype=float64),
     'T': numpy.ndarray(shape=(3, 1), dtype=float64),
     'P': numpy.ndarray(shape=(3, 3), dtype=float64)
    }

It is a dictionary with following keys:

* **K** --- Intrinsic matrix
* **R** --- Rotation matrix
* **T** --- Translation vector
* **P** --- Camera projection matrix


Example
--------------------------

.. code-block:: python3

    from calibration import Calibration

    # initialize keypoints positions
    objects = ...

    c = Calibration().calibrate(objects)
