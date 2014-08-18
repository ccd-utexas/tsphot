#!/usr/bin/env python
"""Utilities for pipelining time-series photometry.

See Also
--------
read_spe : Module for reading SPE files. 

Notes
-----
Conventions : This module's documentation follows the `numpy` doc example [1]_.
'See Also' : Methods describe their relationships to each other within their
    docstrings under the 'See Also' section. All methods should be connected
    to at least one other method within this module [2]_.
PIPELINE_SEQUENCE_NUMBER : Methods are labeled like semantic versioning within
    their docstrings under the 'Notes' section. The sequence number identifies
    in what order the functions are usually called by higher-level scripts.
    - Major numbers (..., -1.0, 0.0, 1.0, 2.0, ...) identify functions
      that are computation/IO-intensive and/or are critical to the pipeline.
    - Minor numbers (..., x.0.1, x.1, x.1.1, , x.2, ...) identify functions
      that are not computation/IO-intensive, are optional to the pipeline,
      and/or are diagnostic.
    - All functions within this module should have a sequence number since
      they should all have a role in the pipeline [2]_.

TODO
----
Include FITS processing.
Write 'Raises' docstring sections.
Write 'Examples' docstring sections.

References
----------
.. [1] https://github.com/numpy/numpy/blob/master/doc/example.py
.. [2] http://en.wikipedia.org/wiki/Pipeline_(software)

"""

# Forwards compatibility imports.
from __future__ import division, absolute_import, print_function

# Standard library imports.
import os
import sys
import math

# External package imports.
# Grouped procedurally then categorically.
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import scipy
from skimage import feature
import matplotlib.pyplot as plt
import astropy
import ccdproc
import imageutils
import photutils
from photutils.detection import morphology, lacosmic
from astroML import stats as astroML_stats

# Internal package imports.
import read_spe

def create_config(fjson='config.json'):
    """Create configuration file for data reduction.

    Parameters
    ----------

    Returns
    -------

    See Also
    --------

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER: -1.0

    TODO
    ----
    Make json config file for reductions

    References
    ----------
    
    """
    pass

def spe_to_dict(fpath):
    """Load an SPE file into a ``dict`` of `ccdproc.CCDData` with metadata.

    Parameters
    ----------
    fpath : string
        Path to SPE file [1]_.

    Returns
    -------
    object_ccddata : dict
        ``dict`` with `ccdproc.CCDData`. Per-frame metadata is stored as `ccdproc.CCDData.meta`.
        SPE file footer is stored under `object_ccddata['footer_xml']`.

    See Also
    --------
    create_master_calib : Next step in pipeline. Run `spe_to_dict` then use the output
        in the input to `create_master_calib`.
    read_spe : Module for reading SPE files.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 0.0

    TODO
    ----
    Return SPE header as well.
    
    References
    ----------
    .. [1] Princeton Instruments SPE 3.0 File Format Specification
           ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/SPE%203.0%20File%20Format%20Specification.pdf

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
    """Create a master calibration frame from a ``dict`` of `ccdproc.CCDData`.
    Median-combine individual calibration frames and retain all metadata.

    Parameters
    ----------
    dobj : dict with ccdproc.CCDData
        ``dict`` keys with non-`ccdproc.CCDData` values are retained as metadata.

    Returns
    -------
    ccddata : ccdproc.CCDData
        A single master calibration frame.
        For `dobj` keys with non-`ccdproc.CCDData` values, the values
        are returned in `ccddata.meta` under the same keys.
        For `dobj` keys with `ccdproc.CCDData` values, the `dobj[key].meta` values
        are returned  are returned as a ``dict`` of metadata.

    See Also
    --------
    spe_to_dict : Previous step in pipeline. Run `spe_to_dict` then use the output
        in the input to `create_master_calib`.
    reduce_ccddata : Next step in pipeline. Run `create_master_calib` to create master
        bias, dark, flat calibration frames and input to `reduce_ccddata`.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 1.0

    References
    ----------
    
    """
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

def gain_readnoise_from_master(bias, flat):
    """Calculate the gain and readnoise from a master bias frame
    and a master flat frame.

    Parameters
    ----------
    bias : array_like
        2D array of a master bias frame.
    flat : array_like
        2D array of a master flat frame.

    Returns
    -------
    gain : float
        Gain of the camera in electrons/ADU.
    readnoise : float
        Readout noise of the camera in electrons/pixel.
    
    See Also
    --------
    reduce_

    Notes
    -----

# from sec 3.7 Overscan and bias, Howell
# relation_readnoise_gain = readnoise / gain
# from sec 4.3. Calculation of read noise and gain, Howell
# relation_flatmean_gain = sqrt(flat_mean * gain) / gain
# Solving for gain and readnoise:
# gain = flat_mean / relation_flatmean_gain**2
# readnoise = gain * relation_readnoise_gain
# Using the median as a more robust estimator of the mean given outliers.
# Using sigmaG as a more robust estimator of distribution sigma given outliers.
# from astroml book
    
    """
    bias_sigmaG = stats.sigmaG(master_bias)
    bias_fwhm   = sigma_to_fwhm(master_bias_sigmaG)
    master_flat_sigmaG = stats.sigmaG(master_flat)
    master_flat_fwhm   = sigma_to_fwhm(master_flat_sigmaG)
    master_flat_median = np.median(master_flat)
    rel_flat_gain = master_flat_fwhm
    rel_readnoise_gain = master_bias_fhwm 
    gain = master_flat_median / rel_flat_gain**2
    readnoise = gain_master * rel_readnoise_gain
    return (gain, readnoise)

def get_exptime_prog(spe_footer_xml):
    """Get the programmed exposure time in seconds from
    the string XML footer of an SPE file.

    Parameters
    ----------
    spe_foooter_xml : string
        ``string`` must be properly formatted XML from a
        Princeton Instruments SPE file footer [1]_.

    Returns
    -------
    exptime_prog_sec : float
        Programmed exposure time in seconds (i.e. the input exposure time from the observer).

    See Also
    --------
    reduce_ccddata : Requires exposure times for frames.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 1.2
    Method uses `bs4.BeautifulSoup` to parse the XML ``string``.
    Converts exposure time to seconds from 'ExposureTime' and 'DelayResolution' XML keywords.

    References
    ----------
    .. [1] Princeton Instruments SPE 3.0 File Format Specification
           ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/SPE%203.0%20File%20Format%20Specification.pdf
    
    """
    footer_xml = BeautifulSoup(spe_footer_xml, 'xml')
    exptime_prog = int(footer_xml.find(name='ExposureTime').contents[0])
    exptime_prog_res = int(footer_xml.find(name='DelayResolution').contents[0])
    exptime_prog_sec =  (exptime_prog / exptime_prog_res)
    return exptime_prog_sec

def reduce_ccddata(dobj, dobj_exptime=None,
                        bias=None,
                        dark=None, dark_exptime=None,
                        flat=None, flat_exptime=None):
    """Reduce a dict of object data frames using the master calibration frames
    for bias, dark, and flat.

    All frames must be type `ccdproc.CCDData`. Method will do all reductions possible
    with given master calibration frames. Method operates on a ``dict``
    in order to minimize the number of pre-reduction operations:
    `dark` - `bias`, `flat` - `bias`, `flat` - `dark`.
    Requires exposure time (seconds) for object data frames.
    If master dark frame is provided, requires exposure time for master dark frame.
    If master flat frame is provided, requires exposure time for master flat frame.

    Parameters
    ----------
    dobj : dict with ccdproc.CCDData
         ``dict`` keys with non-`ccdproc.CCDData` values are retained as metadata.
    dobj_exptime : {None}, float or int, optional
         Exposure time of frames within `dobj`. All frames must have the same expsosure time.
         Required if `dark` is provided.
    bias : {None}, ccdproc.CCDData, optional
        Master bias frame.
    dark : {None}, ccdproc.CCDData, optional
        Master dark frame. Will be scaled to match exposure time for `dobj` frames and `flat` frame.
    dark_exptime : {None}, float or int, optional
        Exposure time of `dark`. Required if `dark` is provided.
    flat : {None}, ccdproc.CCDData, optional
        Master flat frame.
    flat_exptime : {None}, float or int, optional
        Exposure time of `flat`. Required if `flat` is provided.

    Returns
    -------
    dobj_reduced : dict with ccdproc.CCDData
        `dobj` with `ccdproc.CCDData` frames reduced. ``dict`` keys with non-`ccdproc.CCDData` values
        are also returned in `dobj_reduced`.

    See Also
    --------
    create_master_calib : Previous step in pipeline. Run `create_master_calib` to create master
        bias, dark, flat calibration frames and input to `reduce_ccddata`.
    remove_cosmic_rays : Next step in pipeline. Run `reduce_ccddata` then use the output
        in the input to `remove_cosmic_rays`.
    get_exptime_prog : Get programmed exposure time from an SPE footer XML string.
    
    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 2.0
    Sequence of operations (following sec 4.5, "Basic CCD Reduction" [1]_):
    - subtract master bias from master dark
    - subtract master bias from master flat
    - scale and subract master dark from master flat
    - subtract master bias from each object image
    - scale and subtract master dark from each object image
    - divide each object image by corrected master flat

    TODO
    ----
    - Use logging rather than print.
    
    References
    ----------
    .. [1] Howell, 2006, "Handbook of CCD Astronomy"
    
    """
    # Check input.
    # If there is a `dark`...
    if dark != None:
        # ...but no `dobj_exptime` or `dark_exptime`:
        if ((dobj_exptime == None) or
            (dark_exptime == None)):
            raise IOError("If `dark` is provided, both `dobj_exptime` and `dark_exptime` must also be provided.")
    # If there is a `flat`...
    if flat != None:
        # ...but no `flat_exptime`:
        if flat_exptime == None:
            raise IOError("If `flat` is provided, `flat_exptime` must also be provided.")
    # Note: Modify frames in-place to reduce memory overhead.
    # Operations:
    # - subtract master bias from master dark
    # - subtract master bias from master flat
    # - scale and subract master dark from master flat
    if bias != None:
        if dark != None:
            print("INFO: Subtracting master bias from master dark.")
            dark = ccdproc.subtract_bias(dark, bias)
        if flat != None:
            print("INFO: Subtracting master bias from master flat.")
            flat = ccdproc.subtract_bias(flat, bias)
    if ((dark != None) and
        (flat != None)):
        print("INFO: Subtracting master dark from master flat.")
        flat = ccdproc.subtract_dark(flat, dark,
                                     dark_exposure=dark_exptime,
                                     data_exposure=flat_exptime,
                                     scale=True)
    # Print progress through dict.
    # Operations:
    # - subtract master bias from object image
    # - scale and subtract master dark from object image
    # - divide object image by corrected master flat
	keys_sortedlist = sorted(dobj.keys())
	keys_len = len(keys_sortedlist)
	prog_interval = 0.05
	prog_divs = int(math.ceil(1 / prog_interval))
	key_progress = {}
	for idx in xrange(0, prog_divs+1):
	    progress = (idx / prog_divs)
	    key_idx = int(math.ceil((keys_len - 1) * progress))
	    key = keys_sortedlist[key_idx]
	    key_progress[key] = progress
	print("INFO: Reducing object data.\n"+
	      "  Progress (%):", end=' ')
    for key in sorted(dobj):
        if isinstance(dobj[key], ccdproc.CCDData):
            if bias != None:
                dobj[key] = ccdproc.subtract_bias(dobj[key], bias)
            if dark != None:
                dobj[key] = ccdproc.subtract_dark(dobj[key], dark,
                                                   dark_exposure=dark_exptime,
                                                   data_exposure=dobj_exptime)
            if flat != None:
                dobj[key] = ccdproc.flat_correct(dobj[key], flat)
	    if key in key_progress:
	        print(int(key_progress[key]*100), end=' ')        
    return dobj

def remove_cosmic_rays(image,
                       lacosmicargs=dict(contrast=2.0, cr_threshold=4.5, neighbor_threshold=0.45,
                                         gain=0.85, readnoise=6.1)):
    """Remove cosmic rays from an image.

    Method uses the `photutils` implementation of the LA-Cosmic algorithm [1]_.

    Parameters
    ----------
    image : array_like
        2D array of image.
    lacosmicargs : {dict(contrast=2.0, cr_threshold=4.5, neighbor_threshold=0.45,
                         gain=0.85, readnoise=6.1)}, dict
        ``dict`` of keyword arguments for `photutils.detection.lacosmic` [1]_.
        contrast : {2.0}, float
            Chosen from [1]_, and Fig 4 of [2]_.
        cr_threshold : {4.5}, float
            Chosen from test script referenced in [3]_.
        neighbor_threshold : {0.45}, float
            Chosen from test script referenced in [3]_.
        gain : {0.85}, float
            In electrons/ADU. Default is from typical settings for
            Princeton Instruments ProEM 1024B EMCCD [4]_.
        readnoise : {6.1}, float
            In electrons. Default is from typical settings for
            Princeton Instruments ProEM 1024B EMCCD [4]_.

    Returns
    -------
    image_cleaned : numpy.ndarray
        `image` cleaned of cosmic rays as ``numpy.ndarray``.
    ray_mask : numpy.ndarray of bool
        ``numpy.ndarray`` with same dimensions as `image_cleaned` with only
        ``True``/``False`` values. Pixels where cosmic rays were removed are ``True``.
        
    See Also
    --------
    reduce_ccddata : Previous step in pipeline. Run `reduce_ccddata` then use
        the output in the input to `remove_cosmic_rays`.
    find_stars : Next step in pipeline. Run `remove_cosmic_rays` then use the output
        in the input to `find_stars`.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 3.0
    Use LA-Cosmic algorithm from `photutils` rather than `ccdproc` or `imageutils`
        until `ccdproc` issue #130 is closed [3]_.
    `photutils.detection.lacosmic` is verbose in stdout and stderr.

    TODO
    ----
    Use logging.
    
    References
    ----------
    .. [1] http://photutils.readthedocs.org/en/latest/_modules/photutils/detection/lacosmic.html
    .. [2] van Dokkum, 2001. http://adsabs.harvard.edu/abs/2001PASP..113.1420V
    .. [3] https://github.com/astropy/ccdproc/issues/130
    .. [4] Princeton Instruments Certificate of Performance for ProEM 1024B EMCCDs
           with Traditional Amplifier, 1 MHz readout speed, gain setting #3 (highest).
    
    """
    # `photutils.detection.lacosmic` is verbose.
    (image_cleaned, ray_mask) = lacosmic.lacosmic(image, **lacosmicargs)
    return (image_cleaned, ray_mask)
    
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
    array_normd : numpy.ndarray
        Normalized `array` as ``numpy.ndarray``.

    See Also
    -------
    find_stars : `find_stars` calls `normalize` to normalize images before
        searching for stars.
    
    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 3.1
    `array_normd` = (`array` - median(`array`)) / `sigmaG`
    `sigmaG` = 0.7413(q75(`array`) - q50(`array`))
    q50, q75 = 50th, 75th quartiles (q50 == median)

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
          sec 3.2, "Descriptive Statistics"
    
    """
    array_np = np.array(array)
    median = np.median(array_np)
    sigmaG = astroML_stats.sigmaG(array_np)
    if sigmaG == 0:
        # TODO: use logging. STH, 2014-08-11
        print(("WARNING: sigmaG = 0. Normalized array will be all numpy.NaN"),
              file=sys.stderr)
    array_normd = (array_np - median) / sigmaG
    return array_normd
    
def find_stars(image,
               blobargs=dict(min_sigma=1, max_sigma=1, num_sigma=1, threshold=3)):
    """Find stars in an image and return as a dataframe.
    
    Function normalizes the image [1]_ then uses Laplacian of Gaussian method [2]_ [3]_
    to find star-like blobs. Method can also find extended sources by modifying `blobargs`,
    however this pipeline is taylored for stars.
    
    Parameters
    ----------
    image : array_like
        2D array of image.
    blobargs : {dict(min_sigma=1, max_sigma=1, num_sigma=1, threshold=3)}, optional
        Dict of keyword arguments for `skimage.feature.blob_log` [3]_.
        Because image is normalized, `threshold` is the number of stdandard deviations
        above image median for counts per pixel.
        Example for extended sources:
            `blobargs`=dict(`min_sigma`=1, `max_sigma`=30, `num_sigma`=10, `threshold`=3)
    
    Returns
    -------
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : Integer index labeling each found star.
        Columns:
            `x_pix` : x-coordinate (pixels) of found star.
            `y_pix` : y-coordinate (pixels) of found star (pixels).
            `sigma_pix` : Standard deviation (pixels) of the Gaussian kernel
                that detected the blob (usually 1 pixel).

    See Also
    --------
    remove_cosmic_rays : Previous step in pipeline. Run `remove_cosmic_rays`
        then use the output in the input to `find_stars`.
    center_stars : Next step in pipeline. Run `find_stars` then use the output
        in the input to `center_stars`.
    normalize : `find_stars` calls `normalize` to normalize the image before
        finding stars.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 4.0
    Can generalize to extended sources but for increased execution time.
        Execution times for 256x256 image:
        - For example for extended sources above: 0.33 sec/frame
        - For default above: 0.02 sec/frame
    Use `find_stars` after removing cosmic rays to prevent spurrious sources.
    
    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.2, "Descriptive Statistics"
    .. [2] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    .. [3] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    """
    # Normalize image then find stars. Order by x,y,sigma.
    image_normd = normalize(image)
    stars = pd.DataFrame(feature.blob_log(image_normd, **blobargs),
                         columns=['y_pix', 'x_pix', 'sigma_pix'])
    return stars[['x_pix', 'y_pix', 'sigma_pix']]

def plot_stars(image, stars, radius=3,
               imshowargs=dict(interpolation='none')):
    """Plot detected stars overlayed on image.

    Overlay circles around stars and label.
    
    Parameters
    ----------
    image : array_like
        2D array of image.
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : 1 index label for each star.
        Columns:
            `x_pix` : x-coordinate (pixels) of star.
            `y_pix` : y-coordinate (pixels) of star.
    radius : {3}, optional, float or int
        The radius of the circle around each star in pixels.
    imshowargs : {dict(interpolation='none')}, optional
        ``dict`` of keyword arguments for `matplotlib.pyplot.imshow`.

    Returns
    -------
    None : None
        Displays a plot with labeled stars using `matplotlib.pyplot.imshow`.

    See Also
    --------
    find_stars : Run `find_stars` then use the output in the input to
        `plot_stars` as a diagnostic tool.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 4.1
    
    References
    ----------
    .. [1] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    .. [2] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    """
    (fig, ax) = plt.subplots(1, 1)
    ax.imshow(image, **imshowargs)
    for (idx, x_pix, y_pix) in stars[['x_pix', 'y_pix']].itertuples():
        circle = plt.Circle((x_pix, y_pix), radius=radius,
                            color='yellow', linewidth=1, fill=False)
        ax.add_patch(circle)
        ax.annotate(str(idx), xy=(x_pix, y_pix), xycoords='data',
                    xytext=(0,0), textcoords='offset points',
                    color='yellow', fontsize=12, rotation=0)
    plt.show()

def is_odd(num):
    """Determine if a number is equivalent to an odd integer.

    Parameters
    ----------
    num : float or int

    Returns
    -------
    is_odd : bool

    See Also
    --------
    `center_stars` : `center_stars` calls `is_odd` to check that the
        square subframes extraced around each star have an odd number
        pixels on each side.
    
    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 4.2.0
    Uses `math.fabs`, `math.fmod` rather than abs, % [1]_.
    Allows negative numbers.
    (1 - (1E-13)) evaluates as odd.
    (1 + 0j) raises TypeError.

    References
    ----------
    .. [1] https://docs.python.org/2/library/math.html

    """
    # `==` works for floats and ints.
    is_odd = (math.fabs(math.fmod(num, 2)) == 1)
    return is_odd
    
def subtract_subframe_background(subframe, threshold_sigma=3):
    """Subtract the background intensity from a subframe centered on a source.

    The function estimates the background as the median intensity of pixels
    bordering the subframe (i.e. square aperture photometry). Background sigma
    is also computed from the border pixels. The median + number of selected sigma
    is subtracted from the subframe. Pixels whose original intensity was less
    than the median + sigma are set to 0.

    Parameters
    ----------
    subframe : array_like
        2D array of subframe.
    threshold_sigma : {3}, float or int, optional
        `threshold_sigma` is the number of standard
        deviations above the subframe median for counts per pixel. Pixels with
        fewer counts are set to 0. Uses `sigmaG` [2]_.

    Returns
    -------
    subframe_sub : numpy.ndarray
        Background-subtracted `subframe` as ``numpy.ndarray``.

    See Also
    --------
    center_stars : `center_stars` calls `subtract_subframe_background` to
        preprocess subframes around stars for centroid fitting algorithms.

    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 4.2.1
    The source must be centered to within ~ +/- 1/4 of the subframe width.
    At least 3 times as many border pixels used in estimating the background
        as compared to the source [1]_.
    `sigmaG` = 0.7413(q75(`subframe`) - q50(`subframe`))
    q50, q75 = 50th, 75th quartiles (q50 == median)

    References
    ----------
    .. [1] Howell, 2006, "Handbook of CCD Astronomy", sec 5.1.2, "Estimation of Background"
    .. [2] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.2, "Descriptive Statistics"
        
    """
    subframe_np = np.array(subframe)
    (height, width) = subframe_np.shape
    if width != height:
        raise IOError(("Subframe must be square.\n"+
                       "  width = {wid}\n"+
                       "  height = {ht}").format(wid=width,
                                                 ht=height))
    # Choose border width such ratio of number of background pixels to source pixels is >= 3.
    border = int(math.ceil(width / 4))
    arr_longtop_longbottom = np.append(subframe_np[:border],
                                       subframe_np[-border:])
    arr_shortleft_shortright = np.append(subframe_np[border:-border, :border],
                                         subframe_np[border:-border, -border:])
    arr_background = np.append(arr_longtop_longbottom,
                               arr_shortleft_shortright)
    arr_source = subframe_np[border:-border, border:-border]
    if (arr_background.size / arr_source.size) < 3:
        # Howell, 2006, "Handbook of CCD Astronomy", sec 5.1.2, "Estimation of Background"
        raise AssertionError(("Program error. There must be at least 3 times as many sky pixels\n"+
                              "  as source pixels to accurately estimate the sky background level.\n"+
                              "  arr_background.size = {nb}\n"+
                              "  arr_source.size = {ns}").format(nb=arr_background.size,
                                                                 ns=arr_source.size))
    median = np.median(arr_background)
    sigmaG = astroML_stats.sigmaG(arr_background)
    subframe_sub = subframe_np - (median + threshold_sigma*sigmaG)
    subframe_sub[subframe_sub < 0.0] = 0.0
    return subframe_sub

def sigma_to_fwhm(sigma):
    """Convert the standard deviation sigma of a Gaussian into
    the full width at half maximum (FWHM).

    Parameters
    ----------
    sigma : float or int

    Returns
    -------
    fwhm : float
        FWHM = 2*sqrt(2*ln(2))*sigma [1]_.

    See Also
    --------
    center_stars : Within `center_stars`, the centroid fitting method that
        maximizes the flux yielded from an aperture calls `sigma_to_fwhm`.
        
    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 4.2.2
            
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Full_width_at_half_maximum
    
    """
    fwhm = 2.0*math.sqrt(2.0*math.log(2.0))*sigma
    return fwhm

def center_stars(image, stars, box_sigma=11, threshold_sigma=3, method='fit_2dgaussian'):
    """Compute centroids of pre-identified stars in an image and return as a dataframe.

    Extract a square subframe around each star. Side-length of the subframe box is `box_sigma`*`sigma_pix`.
    Subtract the background from the subframe and set pixels with fewer counts than the threshold to 0.
    With the given method, return a dataframe with sub-pixel coordinates of the centroid and
    sigma standard deviation.

    Parameters
    ----------
    image : array_like
        2D array of image.
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : 1 index label for each star.
        Columns:
            `x_pix` : x-coordinate (pixels) of star.
            `y_pix` : y-coordinate (pixels) of star.
            `sigma_pix` : Standard deviation (pixels) of a rough 2D Gaussian fit to the star (usually 1 pixel).
    box_sigma : {11}, int, float, optional
        `box_sigma`*`sigma` x `box_sigma`*`sigma` are the dimensions for a square subframe around the source.
        `box_sigma`*`sigma` will be corrected to be odd and >= 3 so that the center pixel of the subframe is
        the initial `x_pix`, `y_pix`. `box_sigma` is used rather than a fixed box in pixels in order to
        accomodate extended sources. Fitting methods converge to within agreement by `box_sigma` = 11.
    threshold_sigma : {3}, float or int, optional
        `threshold_sigma` is the number of standard deviations above the subframe median
        for counts per pixel. Pixels with fewer counts are set to 0. Uses `sigmaG` [3]_.
    method : {fit_2dgaussian, fit_bivariate_normal}, optional
        The method by which to compute the centroids and sigma.
        `fit_2dgaussian` : Method is from photutils [1]_ and astropy [2]_. Return the centroid coordinates and
            standard devaition sigma from fitting a 2D Gaussian to the intensity distribution. `fit_2dgaussian`
            executes quickly, agrees with `fit_bivariate_normal`, and converges within agreement
            by `box_sigma` = 11. See example below.
        `fit_bivariate_normal` : Model the photon counts within each pixel of the subframe as from a uniform
            distribution [3]_. Return the centroid coordinates and standard deviation sigma from fitting
            a bivariate normal (Gaussian) distribution to the modeled the photon count distribution [4]_.
            `fit_bivariate_sigma` is statistically robust and converges by `box_sigma`= 11, but it executes slowly.
            See example below.
        
    Returns
    -------
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : (same as input `idx`).
        Columns:
            `x_pix` : Sub-pixel x-coordinate (pixels) of centroid.
            `y_pix` : Sub-pixel y-coordinate (pixels) of centroid.
            `sigma_pix` : Sub-pixel standard deviation (pixels) of a 2D Gaussian fit to the star.

    See Also
    --------
    find_stars : Previous step in pipeline. Run `find_stars` then use the output of `find_stars`
        in the input to `center_stars`.
            
    Notes
    -----
    PIPELINE_SEQUENCE_NUMBER : 5.0
    Example: Fitting methods `fit_2dgaussian` and `fit_bivariate_normal` were tested on a bright star with peak
        18000 ADU above background, FHWM ~3.8 pix, initial `sigma_pix` = 1, `box_sigma` = 3 to 33. 2014-08-11, STH. 
        For `fit_2dgaussian`:
        - For varying subframes, position converges to within 0.01 pix of final solution at 11x11 subframe.
        - For varying subframes, sigma converges to within 0.05 pix of final solution at 11x11 subframe.
        - Final position solution agrees with `fit_bivariate_normal` final position solution within +/- 0.1 pix.
        - Final sigma solution agrees with `fit_bivariate_normal` final sigma solution within +/- 0.2 pix.
        - For 11x11 subframe, method takes ~25 ms. Method scales \propto box_sigma.
        For `fit_bivariate_normal`:
        - For varying subframes, position converges to within 0.02 pix of final solution at 11x11 subframe.
        - For varying subframes, sigma converges to within 0.1 pix of final solution at 11x11 subframe.
        - Final position solution agrees with `fit_2dgaussian` final position solution within +/- 0.1 pix.
        - Final sigma solution agrees with `fit_2dgaussian` final sigma solution within +/- 0.2 pix.
        - For 11x11 subframe, method takes ~450 ms. Method scales \propto box_sigma**2. 
    `sigmaG` = 0.7413(q75(`subframe`) - q50(`subframe`))
    q50, q75 = 50th, 75th quartiles (q50 == median)
            
    References
    ----------
    .. [1] http://photutils.readthedocs.org/en/latest/photutils/morphology.html#centroiding-an-object
    .. [2] http://astropy.readthedocs.org/en/latest/api/astropy.modeling.functional_models.Gaussian2D.html
    .. [3] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.3.1., "The Uniform Distribution"
    .. [4] http://www.astroml.org/book_figures/chapter3/fig_robust_pca.html
    
    """
    # Check input.
    valid_methods = ['fit_2dgaussian', 'fit_bivariate_normal']
    if method not in valid_methods:
        raise IOError(("Invalid method: {meth}\n"+
                       "Valid methods: {vmeth}").format(meth=method, vmeth=valid_methods))
    # Make square subframes and compute centroids and sigma by chosed method.
    # Each star or extende source may have a different sigma. Store results in a dataframe.
    stars_init = stars.copy()
    stars_finl = stars.copy()
    stars_finl[['x_pix','y_pix','sigma_pix']] = np.NaN
    for (idx, x_init, y_init, sigma_init) in stars_init[['x_pix', 'y_pix', 'sigma_pix']].itertuples():
        width = int(math.ceil(box_sigma*sigma_init))
        if width < 3:
            width = 3
        if not is_odd(width):
            width = width + 1
        height = width
        # Note:
        # - Subframe may be shortened due to proximity to frame edge.
        # - width, height order is reverse of position x, y order
        # - numpy.ndarrays are ordered by row_idx (y) then col_idx (x)
        # - (0,0) is in upper left.
        subframe = imageutils.extract_array_2d(array_large=image,
                                               shape=(height, width),
                                               position=(x_init, y_init))
        # Compute the initial position for the star relative to the subframe.
        # The initial position relative to the subframe is an integer pixel.
        # If the star was too close to the frame edge to extract the subframe, skip the star.
        (height_actl, width_actl) = subframe.shape
        if ((width_actl == width) and
            (height_actl == height)):
            x_init_sub = (width_actl - 1) / 2
            y_init_sub = (height_actl - 1) / 2
        else:
            # TODO: log events. STH, 2014-08-08
            print(("ERROR: Star is too close to the edge of the frame. Square subframe could not be extracted.\n"+
                   "  idx = {idx}\n"+
                   "  (x_init, y_init) = ({x_init}, {y_init})\n"+
                   "  sigma_init = {sigma_init}\n"+
                   "  box_sigma = {box_sigma}\n"+
                   "  (width, height) = ({width}, {height})\n"+
                   "  (width_actl, height_actl) = ({width_actl}, {height_actl})").format(idx=idx,
                                                                                         x_init=x_init, y_init=y_init,
                                                                                         sigma_init=sigma_init,
                                                                                         box_sigma=box_sigma,
                                                                                         width=width, height=height,
                                                                                         width_actl=width_actl, height_actl=height_actl),
                                                                                         file=sys.stderr)
            continue
        # Compute the centroid position and standard deviation sigma for the star relative to the subframe.
        # using the selected method. Subtract background to fit counts only belonging to the source.
        subframe = subtract_subframe_background(subframe, threshold_sigma)
        if method == 'fit_2dgaussian':
            # Test results: 2014-08-11, STH
            # - Test on star with peak 18k ADU counts above background; FWHM ~3.8 pix.
            # - For varying subframes, position converges to within 0.01 pix of final solution at 11x11 subframe.
            # - For varying subframes, sigma converges to within 0.05 pix of final solution at 11x11 subframe.
            # - Final position solution agrees with `fit_bivariate_normal` final position solution within +/- 0.1 pix.
            # - Final sigma solution agrees with `fit_bivariate_normal` final sigma solution within +/- 0.2 pix.
            # - For 11x11 subframe, method takes ~25 ms. Method scales \propto box_sigma.
            # Method description:
            # - See photutils [1]_ and astropy [2]_.
            # - To calculate the standard deviation for the 2D Gaussian:
            #   zvec = xvec + yvec
            #   xvec, yvec made orthogonal after PCA ('x', 'y' no longer means x,y pixel coordinates)
            #   ==> |zvec| = |xvec + yvec| = |xvec| + |yvec|
            #       Notation: x = |xvec|, y = |yvec|, z = |zvec|
            #   ==> Var(z) = Var(x + y)
            #              = Var(x) + Var(y) + 2*Cov(x, y)
            #              = Var(x) + Var(y)
            #                since Cov(x, y) = 0 due to orthogonality.
            #   ==> sigma(z) = sqrt(sigma_x**2 + sigma_y**2)
            fit = morphology.fit_2dgaussian(subframe)
            (x_finl_sub, y_finl_sub) = (fit.x_mean, fit.y_mean)
            sigma_finl_sub = math.sqrt(fit.x_stddev**2.0 + fit.y_stddev**2.0)
        elif method == 'fit_bivariate_normal':
            # Test results: 2014-08-11, STH
            # - Test on star with peak 18k ADU counts above background; FWHM ~3.8 pix.
            # - For varying subframes, position converges to within 0.02 pix of final solution at 11x11 subframe.
            # - For varying subframes, sigma converges to within 0.1 pix of final solution at 11x11 subframe.
            # - Final position solution agrees with `fit_2dgaussian` final position solution within +/- 0.1 pix.
            # - Final sigma solution agrees with `fit_2dgaussian` final sigma solution within +/- 0.2 pix.
            # - For 11x11 subframe, method takes ~450 ms. Method scales \propto box_sigma**2.
            # Method description:
            # - Model the photons hitting the pixels of the subframe and
            #   robustly fit a bivariate normal distribution.
            # - Conservatively assume that photons hit each pixel, even those of the star,
            #   with a uniform distribution. See [3]_, [4]_.
            # - Seed the random number generator only once per call to this method for reproducibility.
            # - To calculate the standard deviation for the 2D Gaussian:
            #   zvec = xvec + yvec
            #   xvec, yvec made orthogonal after PCA ('x', 'y' no longer means x,y pixel coordinates)
            #   ==> |zvec| = |xvec + yvec| = |xvec| + |yvec|
            #       Notation: x = |xvec|, y = |yvec|, z = |zvec|
            #   ==> Var(z) = Var(x + y)
            #              = Var(x) + Var(y) + 2*Cov(x, y)
            #              = Var(x) + Var(y)
            #                since Cov(x, y) = 0 due to orthogonality.
            #   ==> sigma(z) = sqrt(sigma_x**2 + sigma_y**2)
            x_dist = []
            y_dist = []
            (height_actl, width_actl) = subframe.shape
            np.random.seed(0)
            for y_idx in xrange(height_actl):
                for x_idx in xrange(width_actl):
                    pixel_counts = np.rint(subframe[y_idx, x_idx])
                    x_dist_pix = scipy.stats.uniform(x_idx - 0.5, 1)
                    x_dist.extend(x_dist_pix.rvs(pixel_counts))
                    y_dist_pix = scipy.stats.uniform(y_idx - 0.5, 1)
                    y_dist.extend(y_dist_pix.rvs(pixel_counts))
            (mu, sigma1, sigma2, alpha) = astroML_stats.fit_bivariate_normal(x_dist, y_dist, robust=True)
            (x_finl_sub, y_finl_sub) = mu
            sigma_finl_sub = math.sqrt(sigma1**2.0 + sigma2**2.0)
        # # NOTE: 2014-08-10, STH
        # # The following methods have been commented out because they do not provide an estimate for the star's
        # # standard deviation as a 2D Gaussian.
        # # elif method == 'centroid_com':
        #     # `centroid_com` : Method is from photutils [1]_. Return the centroid from computing the image moments.
        #     # Method is very fast but only accurate between 7 <= `box_sigma` <= 11 given `sigma`=1 due to
        #     # sensitivity to outliers.
        #     # Test results: 2014-08-09, STH
        #     # - Test on star with peak 18k ADU counts above background; platescale = 0.36 arcsec/superpix;
        #     #   seeing = 1.4 arcsec.
        #     # - For varying subframes, method does not converge to final centroid solution.
        #     # - For 7x7 to 11x11 subframes, centroid solution agrees with centroid_2dg centroid solution within
        #     #   +/- 0.01 pix, but then diverges from solution with larger subframes. Method is susceptible to outliers.
        #     # - For 7x7 subframes, method takes ~3 ms per subframe. Method is invariant to box_sigma and alwyas takes ~3 ms.
        #     (x_finl_sub, y_finl_sub) = morphology.centroid_com(subframe)
        # elif method == 'fit_max_phot_flux':
        #     # `fit_max_phot_flux` : Method is from Mike Montgomery, UT Austin, 2014. Return the centroid from computing the
        #     # centroid that yields the largest photometric flux. Method is fast, but, as of 2014-08-08 (STH), implementation
        #     # is inaccurate by ~0.1 pix (given `sigma`=1, `box_sigma`=7), and method is possibly sensitive to outliers.
        #     # Test results: 2014-08-09, STH
        #     # - Test on star with peak 18k ADU counts above background; platescale = 0.36 arcsec/superpix;
        #     #   seeing = 1.4 arcsec.
        #     # - For varying subframes, method converges to within +/- 0.0001 pix of final centroid solution at 7x7 subframe,
        #     #   however final centoid solution disagrees with other methods' centroid solutions.
        #     # - For 7x7 subframe, centroid solution disagrees with centroid_2dg centroid solution for 7x7 subframe
        #     #   by ~0.1 pix. Method may be susceptible to outliers.
        #     # - For 7x7 subframe, method takes ~130 ms. Method scales \propto box_sigma.
        #     # TODO: Test different minimization methods
        #     def obj_func(subframe, position, radius):
        #         """Objective function to minimize: -1*photometric flux from star.
        #         Assumed to follow a 2D Gaussian point-spread function.
        #
        #         Parameters
        #         ----------
        #         subframe : array_like
        #             2D subframe of image. Used only by `obj_func`.
        #         position : list or array of a tuple
        #             Center ``tuple`` coordinate of the aperture within a ``list`` or ``array``,
        #                 i.e. [x_pix, y_pix] [1]_, [2]_.
        #             Used by both `obj_func` and `jac_func`.
        #         radius : float
        #             The radius of the aperture [1]_. Used only by `obj_func`.
        #
        #         Returns
        #         -------
        #         flux_neg : float
        #             Negative flux computed by photutils [2]_.
        #
        #         References
        #         ----------
        #         .. [1] http://photutils.readthedocs.org/en/latest/api/photutils.CircularAperture.html#photutils.CircularAperture
        #         .. [2] http://photutils.readthedocs.org/en/latest/api/photutils.aperture_photometry.html#photutils.aperture_photometry
        #         .. [3] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html
        #
        #         """
        #         aperture = ('circular', radius)
        #         (flux_table, aux_dict) = photutils.aperture_photometry(subframe, position, aperture)
        #         flux_neg = -1. * flux_table['aperture_sum'].data
        #         return flux_neg
        #
        #     def jac_func(subframe, position, radius, eps=0.005):
        #         """Jacobian of the objective function for fixed radius.
        #         Assumed to follow a 2D Gaussian point-spread function.
        #
        #         Parameters
        #         ----------
        #         subframe : array_like
        #             2D subframe of image. Used only by `obj_func`
        #         position : list or array of a tuple
        #             Center ``tuple`` coordinate of the aperture within a ``list`` or ``array``,
        #                 i.e. [x_pix, y_pix] [1]_, [2]_.
        #             Used by both `obj_func` and `jac_func`.
        #         radius : float
        #             The radius of the aperture [1]_. Used only by `obj_func`.
        #         eps : float
        #             Epsilon value for computing the change in the gradient. Used only by `jac_func`.
        #
        #         Returns
        #         -------
        #         jac : numpy.ndarray
        #             Jacobian of obj_func as ``numpy.ndarray`` [dx, dy].
        #
        #         References
        #         ----------
        #         .. [1] http://photutils.readthedocs.org/en/latest/api/photutils.CircularAperture.html#photutils.CircularAperture
        #         .. [2] http://photutils.readthedocs.org/en/latest/api/photutils.aperture_photometry.html#photutils.aperture_photometry
        #         .. [3] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html
        #
        #         """
        #         try:
        #             [x_pix, y_pix] = position
        #         except ValueError:
        #             raise ValueError(("'position' must have the format [x_pix, y_pix]\n"+
        #                               "  position = {pos}").format(pos=position))
        #         jac = np.zeros(len(position))
        #         fxp1 = obj_func(subframe, (x_pix + eps, y_pix), radius)
        #         fxm1 = obj_func(subframe, (x_pix - eps, y_pix), radius)
        #         fyp1 = obj_func(subframe, (x_pix, y_pix + eps), radius)
        #         fym1 = obj_func(subframe, (x_pix, y_pix - eps), radius)
        #         jac[0] = (fxp1-fxm1)/(2.*eps)
        #         jac[1] = (fyp1-fym1)/(2.*eps)
        #         return jac
        #
        #     position = [x_init_sub, y_init_sub]
        #     radius = sigma_to_fwhm(sigma_init)
        #     res = scipy.optimize.minimize(fun=(lambda pos: obj_func(subframe, pos, radius)),
        #                                   x0=position,
        #                                   method='L-BFGS-B',
        #                                   jac=(lambda pos: jac_func(subframe, pos, radius)),
        #                                   bounds=((0, width), (0, height)))
        #     (x_finl_sub, y_finl_sub) = res.x
        else:
            raise AssertionError(("Program error. Method not accounted for: {meth}").format(meth=method))
        # Compute the centroid coordinates relative to the entire image.
        # Return the dataframe with centroid coordinates and sigma.
        (x_offset, y_offset) = (x_finl_sub - x_init_sub,
                                y_finl_sub - y_init_sub)
        (x_finl, y_finl) = (x_init + x_offset,
                            y_init + y_offset)
        sigma_finl = sigma_finl_sub
        stars_finl.loc[idx, ['x_pix', 'y_pix', 'sigma_pix']] = (x_finl, y_finl, sigma_finl)
    return stars_finl
