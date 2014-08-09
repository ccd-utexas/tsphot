#!/usr/bin/env python
"""Utilities for time-series photometry.

"""

from __future__ import division, absolute_import, print_function

import math
import inspect

import read_spe
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import astropy
import ccdproc
import imageutils
import photutils
from astroML import stats
from skimage import feature, measure
import matplotlib.pyplot as plt

def create_config(fjson='config.json'):
    """Create configuration file for data reduction.
    
    """
    # TODO: make config file for reductions
    pass

def spe_to_dict(fpath):
    """Load an SPE file into a dict of ccdproc.ccddata.
    
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
    """Create master calibration frame from dict of ccdproc.ccddata.
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
    """Get the programmed exposure time in seconds
    from the string XML footer of an SPE file.
    
    """
    footer_xml = BeautifulSoup(spe_footer_xml, 'xml')
    exptime_prog = int(footer_xml.find(name='ExposureTime').contents[0])
    exptime_prog_res = int(footer_xml.find(name='DelayResolution').contents[0])
    return (exptime_prog / exptime_prog_res)

def reduce_ccddata_dict(dobj, bias=None, dark=None, flat=None,
                        dobj_exptime=None, dark_exptime=None, flat_exptime=None):
    """Reduce a dict of object data frames using the master calibration frames
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

def normalize(array):
    """Normalize an array in a robust way.

    The function flattens an array then normalizes in a way that is 
    insensitive to outliers (i.e. ignore stars on an image of the night sky).
    Following [1]_, the function uses `sigmaG` as a width estimator and
    uses the median as an estimator for the mean.
    
    Parameters
    ----------
    array : array_like
        Array can be flat or nested.

    Returns
    -------
    array_normd : array_like
        Normalized version of `array`.

    Notes
    -----
    `normd_array = (array - median) / sigmaG`
    `sigmaG = 0.7413(q75 - q50)`
    q50, q75 = 50th, 75th quartiles
    See [1]_.

    References
    ----------
     .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
        sec 3.2, "Descriptive Statistics"
    
    """
    sigmaG = stats.sigmaG(array)
    median = np.median(array)
    return (array - median) / sigmaG

def detect_blobs(image,
                 blobargs=dict(min_sigma=1, max_sigma=1, num_sigma=1, threshold=2)):
    """Detect blobs in an image.
    
    Function normalizes the image [1]_ then uses Laplacian of Gaussian method [2]_ [3]_
    to find source-like blobs. Method finds stars and galaxies.
    
    Parameters
    ----------
    image : array_like
        2D array of image.
    blobargs : {dict(min_sigma=1, max_sigma=1, num_sigma=1, threshold=2)}, optional
        Dict of keyword arguments for `skimage.feature.blob_log` [3]_.
        Can generalize to extended sources but for reduced performance
        On 256x256 image: for example below, 0.33 sec/frame; for default above, 0.02 sec/frame.
        Because image is normalized, `threshold` is the number of stdandard deviations
        above background for counts per pixel.
        Example: `blobargs=dict(min_sigma=1, max_sigma=30, num_sigma=10, threshold=2)`
    
    Returns
    -------
    blobs : pandas.DataFrame
        ``pandas.DataFrame`` with 1 row for each blob and columns `x_blob`', `y_blob`, `sigma_blob`.
        `sigma_blob` is the standard deviation of the Gaussian kernel that detected the blob (usually 1 pixel).
    
    Notes
    -----
    Blob radius is ~`sqrt(2)*sigma_blob` [3]_.
    Blob radius and `sigma_blob` are not robust estimators of seeing FWHM.
    
    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
        sec 3.2, "Descriptive Statistics"
    .. [2] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    .. [3] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    """
    image_normd = normalize(image)
    blobs = pd.DataFrame(feature.blob_log(image_normd, **blobargs), columns=['y_blob', 'x_blob', 'sigma_blob'])
    return blobs[['x_blob', 'y_blob', 'sigma_blob']]

def plot_detected_blobs(image, blobs):
    """Plot detected blobs overlayed on image.

    Overlay image markers for blobs. Convert blob sigma to radius.
    From skimage gallery [1]_.
    
    Parameters
    ----------
    image : array_like
        2D array of image.
    blobs : pandas.DataFrame
        ``pandas.DataFrame`` with 1 row for each blob and includes columns `x_blob`, `y_blob`, `sigma_blob`.
        `sigma_blob` is the standard deviation of the Gaussian kernel that detected the blob (usually 1 pixel).

    Returns
    -------
    None
    
    Notes
    -----
    Blob radius is ~`sqrt(2)*sigma_blob` [2]_.
    Blob radius and `sigma_blob` are not robust estimators of seeing FWHM.
    
    References
    ----------
    .. [1] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    .. [2] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    """
    (fig, ax) = plt.subplots(1, 1)
    ax.imshow(image, interpolation='nearest')
    for (idx, x_blob, y_blob, sigma_blob) in blobs[['x_blob', 'y_blob', 'sigma_blob']].itertuples():
        radius_blob = sigma_blob*math.sqrt(2)
        c = plt.Circle((x_blob, y_blob), radius_blob, color='yellow', linewidth=2, fill=False)
        ax.add_patch(c)
    plt.show()

def center_stars(image, blobs, box_sigma=5, method='fit_bivariate_normal'):
    """Find star centroids in an image with identified stars.

    Extract a subframe around each star. Length of the subframe is a factor of star sigma [2]_.
    Use a method to find the centroid. Return the dataframe with new columns for sub-pixel
    x-, y-coordinates of centroid and standard deviation sigma of intensity distribution.

    Parameters
    ----------
    image : array_like
        2D array of image.
    blobs : pandas.DataFrame
        ``pandas.DataFrame`` with 1 row for each blob and includes columns `x_blob`, `y_blob`, `sigma_blob`.
        `sigma_blob` is the standard deviation of the Gaussian kernel that detected the blob (usually 1 pixel).
    box_sigma : {5}, optional
        The dimensions for a subframe around the source for centering are
        `box_sigma*sigma_blob` x `box_sigma*sigma_blob`. `sigma_blob` is usually 1 pixel.
        Number should be odd so that center pixel of subframe is initial `x_blob`, `y_blob`.
    method : {max_flux, daofind, irafstarfind, moments}, optional
        The method by which to compute the centroids.
        `max_flux` : Return the centroid that yields the highest flux from photoemtry (from Mike Montgomery, UT Austin, 2014).
        `moments` : Return the centroid from the image moments [3]_.

    Returns
    -------
    blobs : pandas.DataFrame
        ``pandas.DataFrame`` with 1 row for each blob and added columns `x_ctrd`, `y_ctrd`.
    
    Notes
    -----
    Blob radius is ~`sqrt(2)*sigma_blob` [1]_.
    Blob radius and `sigma_blob` are not robust estimators of seeing FWHM.

    TODO
    ----
    Include Mike's max_phot_flux method, irafstarfind image moments method
    daophot 2D Gaussian method.
    
    References
    ----------
    .. [1] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    .. [2] https://imageutils.readthedocs.org/en/latest/api/imageutils.extract_array_2d.html#imageutils.extract_array_2d
    .. [3] http://scikit-image.org/docs/dev/api/skimage.measure.html#moments
    
    """
    # Check input
    valid_methods = ['fit_bivariate_normal']
    if method not in valid_methods:
        raise IOError(("Invalid method: {meth}\n"+
                       "Valid methods: {vmeth}").format(meth=method, vmeth=valid_methods))
    # Make box subframes and compute centroids per chosed method. Each blob may have a different size.
    # Subframe can be shortened due to proximity to frame edge.
    (blobs['x_ctrd'], blobs['y_ctrd']) = (np.NaN, np.NaN)
    for (idx, x_blob, y_blob, sigma_blob) in blobs[['x_blob', 'y_blob', 'sigma_blob']].itertuples():
        width = np.round(box_sigma*sigma_blob)
        height = width
        # width, height order is reverse of position x, y order
        subframe = imageutils.extract_array_2d(array_large=image,
                                               shape=(height, width),
                                               position=(x_blob, y_blob))
        elif method == 'fit_bivariate_normal':
            # Model the photons hitting the pixels of the subframe and
            # robustly fit a bivariate normal distribution.
            # Photons hit each pixel with a uniform distribution (sec 3.3).
            # http://www.astroml.org/book_figures/chapter3/fig_robust_pca.html

            
            pass
        else:
            raise AssertionError(("Program error. Input method not accounted for: {meth}").format(meth=method))
        # Compute the centroid coordinates relative to the blob centers.
        (x_sub_blob, y_sub_blob) = map(lambda dim: (np.round(dim) - 1)/2, (width, height))
        (x_offset, y_offset) = (x_sub_ctrd - x_sub_blob,
                                y_sub_ctrd - y_sub_blob)
        blobs.loc[idx, ['x_ctrd', 'y_ctrd']] = (x_blob + x_offset,
                                                y_blob + y_offset)
    return blobs

