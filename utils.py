#!/usr/bin/env python
"""
Utilities for time-series photometry.
"""

from __future__ import print_function, division
import os
import sys
import csv
import pickle
import inspect
import astropy
import ccdproc
import read_spe
import collections
import numpy as np
import pandas as pd
import datetime as dt
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

def create_config(fjson='config.json'):
    """
    Create configuration file for data reduction.
    """
    # TODO: make config file for reductions
    pass

def spe_to_dict(fpath):
    """
    Load an SPE file into a dict of ccdproc.ccddata.
    """
    spe = read_spe.File(fpath)
    object_ccddata = {}
    object_ccddata['footer_xml'] = spe.footer_metadata
    for fidx in xrange(spe.get_num_frames()):
        (data, meta) = spe.get_frame(fidx)
        object_ccddata[fidx] = ccdproc.CCDData(data=data, meta=meta, unit=astropy.units.adu)
    spe.close()
    return object_ccddata
    
def create_master_calib(dobj):
    """
    Create master calibration frame from dict of ccdproc.ccddata.
    Median-combine individual calibration frames and retain all metadata.
    """
    # TODO:
    # - Use multiprocessing to side-step global interpreter lock and parallelize.
    #   https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing
    # STH, 20140716
    combiner_list = []
    noncombiner_list = []
    fidx_meta = {}
    for key in dobj:
        # If the key is an index for a CCDData frame...
        if isinstance(dobj[key], ccdproc.CCDData):
            combiner_list.append(dobj[key])
            fidx_meta[key] = dobj[key].meta
        # ...otherwise save it as metadata.
        else:
            noncombiner_list.append(key)
    ccddata = ccdproc.Combiner(combiner_list).median_combine()
    ccddata.meta['fidx_meta'] = fidx_meta
    for key in noncombiner_list:
        ccddata.meta[key] = dobj[key]
    return ccddata

def get_exptime_prog(spe_footer_xml):
    """
    Get the programmed exposure time in seconds
    from the string XML footer of an SPE file.
    """
    footer_xml = BeautifulSoup(spe_footer_xml, 'xml')
    exptime_prog = int(footer_xml.find(name='ExposureTime').contents[0])
    exptime_prog_res = int(footer_xml.find(name='DelayResolution').contents[0])
    return (exptime_prog / exptime_prog_res)

def reduce_ccddata_dict(dobj, bias=None, dark=None, flat=None,
                        dobj_exptime=None, dark_exptime=None, flat_exptime=None):
    """
    Reduce a dict of object data frames using the master calibration frames
    for bias, dark, and flats. All frames must be type ccdproc.CCDData.
    Requires exposure times (seconds) for object data frames, master dark, and master flat.
    Operations (from sec 4.5, Basic CCD Reduction, of Howell, 2006, Handbook of CCD Astronomy):
    - subtract master bias from master dark
    - subtract master bias from master flat
    - scale and subract master dark from master flat
    - subtract master bias from object
    - scale and subtract master dark from object
    - divide object by normalized master flat
    """
    # TODO:
    # - parallelize
    # - Compute and correct ccdgain
    #   STH, 20140805
    # Check input.
    iframe = inspect.currentframe()
    (args, varargs, keywords, ilocals) = inspect.getargvalues(iframe)
    for arg in args:
        if ilocals[arg] == None:
            print(("INFO: {arg} is None.").format(arg=arg))
    # Operations:
    # - subtract master bias from master dark
    # - subtract master bias from master flat
    if bias != None:
        if dark != None:
            dark = ccdproc.subtract_bias(dark, bias)
        if flat != None:
            flat = ccdproc.subtract_bias(flat, bias)
    # Operations:
    # - scale and subract master dark from master flat
    if ((dark != None) and
        (flat != None)):
        flat = ccdproc.subtract_dark(flat, dark,
                                     dark_exposure=dark_exptime,
                                     data_exposure=flat_exptime,
                                     scale=True)
    # Operations:
    # - subtract master bias from object
    # - scale and subtract master dark from object
    # - divide object by normalized master flat
    for fidx in dobj:
        if isinstance(dobj[fidx], ccdproc.CCDData):
            if bias != None:
                dobj[fidx] = ccdproc.subtract_bias(dobj[fidx], bias)
            if dark != None:
                dobj[fidx] = ccdproc.subtract_dark(dobj[fidx], dark,
                                                   dark_exposure=dark_exptime,
                                                   data_exposure=dobj_exptime)
            if flat != None:
                dobj[fidx] = ccdproc.flat_correct(dobj[fidx], flat)
    # Remove cosmic rays
    for fidx in dobj:
        if isinstance(dobj[fidx], ccdproc.CCDData):
            dobj[fidx] = ccdproc.cosmicray_lacosmic(dobj[fidx], thresh=5, mbox=11, rbox=11, gbox=5)
    return dobj

def detect_blobs(image, blobargs=dict(threshold=3)):
    """Detect blobs in an image.
    
    Function normalizes the image [1]_ then uses Laplacian of Gaussian method [2]_ [3]_
    to find source-like blobs. Method finds stars and galaxies.
    
    Parameters
    ----------
    image : array_like
        2D array of image.
    blobargs : {dict(threshold=3)}, optional
        Dict of keyword arguments for `skimage.feature.blob_log` [3]_. Can speed up blob finding by ~25x.
        Because image is normalized, `threshold` is the number of stdandard deviations
        above background for counts per pixel.
        Example: `blobargs=dict(min_sigma=1, max_sigma=1, num_sigma=1, threshold=4)`
    
    Returns
    -------
    blobs : 2D ndarray
        Array with elements `(x-coordinate, y-coordinate, sigma)` for each blob.
        `sigma` is the standard deviation of the Gaussian kernel that detected the blob.
    
    Notes
    -----
    Blob radius is ~`sqrt(2)*sigma` [3]_. `sigma` is not a robust estimator of FWHM.
    
    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
        sec 3.2, "Descriptive Statistics"
    .. [2] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    .. [3] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    """
    # Robustly normalize image using width estimator and median.
    # From [1]
    sigmaG = stats.sigmaG(image_orig)
    median = np.median(image_orig)
    image_norm = (image - median) / sigmaG
    # Find blobs. 'threshold' is number of std. dev. above background for counts per pixel.
    # Change from (y, x, sigma) to (x, y, sigma)
    # From [2]
    blobs = blob_log(image_norm, **blobargs)
    return blobs[:, [1, 0, 2]]

def plot_detected_blobs(image, blobs):
    """Plot detected blobs overlayed on image.
    
    From skimage gallery [1]_.
    
    Parameters
    ----------
    image : array_like
        2D array of image.
    blobs : array_like
        Array with elements `(x-coordinate, y-coordinate, radius)` for each blob.
        
    Returns
    -------
    None
    
    References
    ----------
    .. [1] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    """
    (fig, ax) = plt.subplots(1, 1)
    ax.imshow(image_orig, interpolation='nearest')
    for blob in blobs:
        (x, y, r) = blob
        c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
        ax.add_patch(c)
    plt.show()
