.. _information_label:

Information
=============================================================

This tool contains different methods for automatic surveillance camera calibration.

The implemented are: `OptInOpt <https://ieeexplore.ieee.org/abstract/document/8909905>`_,  `PlaneCalib <https://ieeexplore.ieee.org/document/9363417>`_ and `LandmarkCalib <https://link.springer.com/article/10.1007/s00138-020-01125-x>`_.
These methods serves for camera calibration based on 2D-3D correspondences of some keypoints. Contrary to classical PnP solution 3D keypoints does not lie within common coordinate system.
More information can be found in :ref:`usage_label` or see proper papers.

Requirements
---------------------------

* tensorflow (tensorflow-gpu)
* opencv-python
* scipy

Tensorflow version TF1 and TF2 are acceptable.
Tested on the following installation:

::

    absl-py==0.10.0
    astor==0.8.1
    gast==0.4.0
    google-pasta==0.2.0
    grpcio==1.32.0
    h5py==2.10.0
    importlib-metadata==2.0.0
    Keras-Applications==1.0.8
    Keras-Preprocessing==1.1.2
    Markdown==3.2.2
    numpy==1.19.2
    opencv-python==4.4.0.44
    protobuf==3.13.0
    scipy==1.5.2
    six==1.15.0
    tensorboard==1.14.0
    tensorflow-estimator==1.14.0
    tensorflow-gpu==1.14.0
    termcolor==1.1.0
    Werkzeug==1.0.1
    wrapt==1.12.1
    zipp==3.3.0
