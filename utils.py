#!/usr/bin/env python
"""Utilities for pipelining time-series photometry.

See Also
--------
read_spe, main

Notes
-----
noinspection : Comments are created by PyCharm to flag permitted code inspection violations.
docstrings : This module's documentation is adapted from the `numpy` doc example [1]_.
TODO : Flag to-do items with 'TODO:' in the code body so that they are aggregated when using an IDE.
See Also : Tuple of related objects (modules, classes, methods, etc). Tuple usually includes objects that this object
    calls, objects that call this object, and objects that are functionally related to this object.
SEQUENCE_NUMBER : Methods are labeled like semantic versioning [3]_ within their docstrings under 'Notes'.
    The sequence number identifies in what order the functions are typically called by higher-level scripts.
    Major numbers (..., -1.0, 0.0, 1.0, 2.0, ...) identify functions that are computation/IO-intensive and/or are
        critical to the pipeline.
    Minor numbers (..., x.0.1, x.1, x.1.1, , x.2, ...) identify functions that are not computation/IO-intensive,
        are optional to the pipeline, and/or are diagnostic.
    All functions within this module should have a sequence number since they should all have a role in the
        pipeline [2]_.
IDE comments : Some comments are flags for IDEs (e.g. Eclipse PyDev, Spyder, PyCharm)
    #@UndefinedVariable : Eclipse PyDev does not recognize imports such as scipy.signal.med2d
    # noinspection : PyCharm flag to not inspect for statement or function

References
----------
.. [1] https://github.com/numpy/numpy/blob/master/doc/example.py
.. [2] http://en.wikipedia.org/wiki/Pipeline_(software)
.. [3] http://semver.org/

"""
# TODO: Use sphinx documentation instead: http://sphinx-doc.org/
# TODO: Include FITS processing.

# Forwards compatibility imports.
from __future__ import division, absolute_import, print_function
# Standard library imports.
import os
import sys
import pdb
import math
import json
import logging
import itertools
import collections
import datetime as dt
# External package imports. Grouped procedurally then categorically.
# Note: Must import modules directly from skimage, sklearn, photutils.
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import dateutil
import scipy
import skimage
from skimage import feature
import sklearn.cross_validation as sklearn_cval
import matplotlib.pyplot as plt
import astropy
import ccdproc
import imageutils
import photutils
from photutils.detection import morphology, lacosmic
# noinspection PyPep8Naming
import astroML.stats as astroML_stats
# Internal package imports.
import read_spe


# TODO: def create_logging_config (for logging dictconfig)
# TODO: resolve #noinspection PyUnresolvedReferences


def create_reduce_config(fjson='reduce_config.json'):
    """Create JSON configuration file for data reduction.

    Parameters
    ----------
    fjson : {'reduce_config.json'}, string, optional
        Path to write default JSON configuration file.

    Returns
    -------
    None

    See Also
    --------
    check_reduce_config

    Notes
    -----
    SEQUENCE_NUMBER: -1.0

    """
    # TODO: Describe key, value pairs in docstring.
    # To omit an argument in the config file, set it to `None`.
    config_settings = collections.OrderedDict()
    config_settings['comments'] = ["Insert multiline comments here. For formatting, see http://json.org/",
                                   "Use JSON `null`/`true`/`false` for empty/T/F values.",
                                   "  See example `null` value for ['master']['dark'].",
                                   "For ['logging']['level'], choices are (from most to least verbose):",
                                   "  ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']",
                                   "  See https://docs.python.org/2/library/logging.html#logging-levels"]
    config_settings['logging'] = collections.OrderedDict()
    config_settings['logging']['filename'] = "tsphot.log"
    config_settings['logging']['level'] = "INFO"
    config_settings['calib'] = collections.OrderedDict()
    config_settings['calib']['bias'] = "calib_bias.spe"
    config_settings['calib']['dark'] = "calib_dark.spe"
    config_settings['calib']['flat'] = "calib_flat.spe"
    config_settings['master'] = collections.OrderedDict()
    config_settings['master']['bias'] = "master_bias.pkl"
    config_settings['master']['dark'] = None
    config_settings['master']['flat'] = "master_flat.pkl"
    config_settings['object'] = collections.OrderedDict()
    config_settings['object']['raw'] = "object_raw.spe"
    config_settings['object']['reduced'] = "object_reduced.pkl"
    # Use binary read-write for cross-platform compatibility. Use Python-style indents in the JSON file.
    with open(fjson, 'wb') as fobj:
        json.dump(config_settings, fobj, sort_keys=False, indent=4)
    return None


def check_reduce_config(dobj):
    """Check configuration settings for data reduction.

    Parameters
    ----------
    dobj : dict
        ``dict`` of configuration settings.

    Returns
    -------
    None

    Raises
    ------
    IOError
        Raised when file doesn't exists, file extension is wrong, or keywords are missing.

    See Also
    --------
    create_reduce_config

    Notes
    -----
    SEQUENCE_NUMBER : -0.9

    """
    # Logging file path need not be defined, but if it is then it must be .log.
    fname = dobj['logging']['filename']
    if fname is not None:
        (fbase, ext) = os.path.splitext(os.path.basename(fname))
        if ext != '.log':
            raise IOError("Logging filename extension is not '.log': {fpath}".format(fpath=fname))
    # Logging level must be set and be one of https://docs.python.org/2/library/logging.html#logging-levels
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    level = dobj['logging']['level']
    if level not in valid_levels:
        raise IOError(("Invalid logging level: {level}\n" +
                       "Valid logging levels: {vlevels}").format(level=level, vlevels=valid_levels))
    # Calibration image file paths need not be defined, but if they are then they must exist and be .spe.
    calib_fpath = dobj['calib']
    for imtype in calib_fpath:
        cfpath = calib_fpath[imtype]
        if cfpath is not None:
            if not os.path.isfile(cfpath):
                raise IOError("Calibration image file does not exist: {fpath}".format(fpath=cfpath))
            (fbase, ext) = os.path.splitext(os.path.basename(cfpath))
            if ext != '.spe':
                raise IOError("Calibration image file extension is not '.spe': {fpath}".format(fpath=cfpath))
    # Master calibration image file paths need not be defined, but if they are then they must be .pkl.
    master_fpath = dobj['master']
    for imtype in master_fpath:
        mfpath = master_fpath[imtype]
        if mfpath is not None:
            (fbase, ext) = os.path.splitext(os.path.basename(mfpath))
            if ext != '.pkl':
                raise IOError("Master calibration image file extension is not '.pkl': {fpath}".format(fpath=mfpath))
    # All calibration image image types must have a corresponding type for master image,
    # even if the file path is not defined.
    for imtype in calib_fpath:
        if imtype not in master_fpath:
            raise IOError(("Calibration image type is not in master calibration image types.\n" +
                           "calibration image type: {imtype}\n" +
                           "master image types: {imtypes}").format(imtype=imtype,
                                                                   imtypes=master_fpath.keys()))
    # Raw object image file path must exist and must be .spe.
    rawfpath = dobj['object']['raw']
    if (rawfpath is None) or (not os.path.isfile(rawfpath)):
        raise IOError("Raw object image file does not exist: {fpath}".format(fpath=rawfpath))
    (fbase, ext) = os.path.splitext(os.path.basename(rawfpath))
    if ext != '.spe':
        raise IOError("Raw object image file extension is not '.spe': {fpath}".format(fpath=rawfpath))
    # Reduced object image file path need not be defined, but if it is then it must be .pkl.
    redfpath = dobj['object']['reduced']
    if redfpath is not None:
        (fbase, ext) = os.path.splitext(os.path.basename(redfpath))
        if ext != '.pkl':
            raise IOError("Reduced object image file extension is not '.pkl': {fpath}".format(fpath=redfpath))
    return None


# ######################################################################################################################
# CREATE LOGGER and other global variables
# Note: Use logger only after checking configuration file.
# Note: For non-root-level loggers, use `getLogger(__name__)`
# http://stackoverflow.com/questions/17336680/python-logging-with-multiple-modules-does-not-work
logger = logging.getLogger(__name__)
# Maximum sigma (in pixels) for Gaussian kernel used for finding stars, dropping duplicates, and matching stars.
# max_sigma = 5.0 for very poor conditions.
# TODO: make class to manage max_sigma variable if need to change. (Bad practice to modify global vars.)
max_sigma = 5.0


def define_progress(dobj, interval=0.05):
    """Return a function (a closure) that prints the progress through a ``dict`` with ``ccdproc.CCDData``.

    The returned function must be called when iterating through the sorted keys of the ``dict``.

    Parameters
    ----------
    dobj : dict
        ``dict`` with ``ccdproc.CCDData``.
    interval : {0.05}, float, optional
        Increments at which to print progress.
        Example: `interval`=0.05 will print progress at keys that correspond to
            0%, 5%, 10%, ..., 95%, 100% through ``dict``.

    Returns
    -------
    print_progress : function
        Call `print_progress` during iteration through sorted keys of ``dict``.

    See Also
    --------
    logger

    Notes
    -----
    SEQUENCE_NUMBER = -0.8

    Examples
    --------
    ```
    print_progress = define_progress(dobj=dobj)
    for key in sorted(dobj):
        print_progress(key=key)
    ```

    """
    # TODO: make isinstance test an argument to generalize to other data (e.g. HDF5, FITS).
    # Check input.
    if (interval <= 0.0) or (interval > 1.0):
        raise IOError(("`interval` must be > 0.0 and <= 1.0:\ninterval = {interval}").format(interval=interval))
    # Define keys to track and create function to print progress.
    image_keys = sorted([key for key in dobj.keys() if isinstance(dobj[key], ccdproc.CCDData)])
    num_keys = len(image_keys)
    divisions = int(math.ceil(1.0 / interval))
    key_progress = {}
    for div_idx in xrange(0, divisions + 1):
        progress = (div_idx / divisions)
        key_idx = int(math.ceil((num_keys - 1) * progress))
        key = image_keys[key_idx]
        key_progress[key] = progress
    # noinspection PyShadowingNames
    def print_progress(key, key_progress=key_progress):
        if key in key_progress:
            logger.info("Progress (%): {pct}".format(pct=int(key_progress[key] * 100)))

    logger.debug("Progress: total number images = {num}, percent interval = {intvl}".format(num=num_keys,
                                                                                            intvl=interval))
    return print_progress


# noinspection PyDictCreation
def spe_to_dict(fpath):
    """Load an SPE file into a ``dict`` of `ccdproc.CCDData` with metadata.

    Parameters
    ----------
    fpath : string
        Path to SPE file [1]_.

    Returns
    -------
    object_ccddata : dict
        ``dict`` with `ccdproc.CCDData`. Per-image metadata is stored as `ccdproc.CCDData.meta`.
        SPE file footer is stored under `object_ccddata['footer_xml']`.

    See Also
    --------
    create_reduce_config, create_master_calib, read_spe

    Notes
    -----
    SEQUENCE_NUMBER : 0.0

    References
    ----------
    .. [1] Princeton Instruments SPE 3.0 File Format Specification
           ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/
           SPE%203.0%20File%20Format%20Specification.pdf

    """
    # TODO : Return SPE header as well.
    # TODO: use hdf5
    spe = read_spe.File(fpath)
    object_ccddata = {}
    object_ccddata['footer_xml'] = spe.footer_metadata
    for fidx in xrange(spe.get_num_frames()):
        (data, meta) = spe.get_frame(fidx)
        object_ccddata[fidx] = ccdproc.CCDData(data=data, meta=meta, unit=astropy.units.adu) #@UndefinedVariable
    spe.close()
    return object_ccddata


def create_master_calib(dobj):
    """Create a master calibration image from a ``dict`` of `ccdproc.CCDData`.

    Median-combine individual calibration images and retain all metadata.

    Parameters
    ----------
    dobj : dict
        ``dict`` with ``ccdproc.CCDData`` values. Non-``ccdproc.CCDData`` values are retained as metadata.

    Returns
    -------
    ccddata : ccdproc.CCDData
        A single master calibration image.
        For `dobj` keys with non-``ccdproc.CCDData`` values, the values are returned in ``ccddata.meta`` under
        the same keys. For `dobj` keys with ``ccdproc.CCDData`` values, the `dobj[key].meta` values are returned
        are returned as a ``dict`` of metadata.

    See Also
    --------
    spe_to_dict, reduce_ccddata

    Notes
    -----
    SEQUENCE_NUMBER : 1.0

    """
    combiner_list = []
    noncombiner_list = []
    fidx_meta = {}
    for key in dobj:
        # If the key is an index for a CCDData image...
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


def sigma_to_fwhm(sigma):
    """Convert the standard deviation sigma of a Gaussian into the full-width-at-half-maximum (FWHM).

    Parameters
    ----------
    sigma : float

    Returns
    -------
    fwhm : float
        fwhm = 2*sqrt(2*ln(2))*sigma [1]_.

    See Also
    --------
    center_stars : Within `center_stars`, the centroid fitting method that
        maximizes the flux yielded from an aperture calls `sigma_to_fwhm`.

    Notes
    -----
    SEQUENCE_NUMBER : 1.0.1

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Full_width_at_half_maximum

    """
    fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma
    return fwhm


# noinspection PyPep8Naming
def gain_readnoise_from_master(bias, flat):
    """Calculate the gain and readnoise from a master bias image and a master flat image.

    Parameters
    ----------
    bias : numpy.ndarray
        2D ``numpy.ndarray`` of a master bias image.
    flat : numpy.ndarray
        2D ``numpy.ndarray`` of a master flat image.

    Returns
    -------
    gain : float
        Gain of the camera in electrons/ADU.
    readnoise : float
        Readout noise of the camera in electrons.

    See Also
    --------
    create_master_calib, gain_readnoise_from_random

    Notes
    -----
    SEQUENCE_NUMBER = 1.1
    from [1]_:
        fwhm_bias = readnoise / gain
    from [2]_:
        fwhm_flat = sqrt(mean_flat * gain) / gain
    Solving for gain and readnoise:
        gain = mean_flat / fwhm_flat**2
        readnoise = gain * fwhm_bias
    from [3]_:
        Using the median as estimator of average because robust to outliers.
        Using sigmaG as estimator of standard deviation because robust to outliers.

    References
    ----------
    .. [1] Howell, 2006, "Handbook of CCD Astronomy", sec 3.7 "Overscan and bias"
    .. [2] Howell, 2006, "Handbook of CCD Astronomy", sec 4.3 "Calculation of read noise and gain"
    .. [3] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
        sec 3.2, "Descriptive Statistics"

    """
    # TODO: As of 2014-08-18, gain_readnoise_from_master and gain_readnoise_from_random do not agree. Check formulas.
    sigmaG_bias = astroML_stats.sigmaG(bias)
    fwhm_bias = sigma_to_fwhm(sigmaG_bias)
    sigmaG_flat = astroML_stats.sigmaG(flat)
    fwhm_flat = sigma_to_fwhm(sigmaG_flat)
    median_flat = np.nanmedian(flat)
    gain = (median_flat / (fwhm_flat ** 2.0))
    readnoise = (gain * fwhm_bias)
    return (gain * (astropy.units.electron / astropy.units.adu), readnoise * astropy.units.electron) #@UndefinedVariable


# noinspection PyPep8Naming
def gain_readnoise_from_random(bias1, bias2, flat1, flat2):
    """Calculate gain and readnoise from a pair of random bias images and a pair of random flat images.

    Parameters
    ----------
    bias1 : numpy.ndarray
        2D ``numpy.ndarray`` of bias image 1.
    bias2 : numpy.ndarray
        2D ``numpy.ndarray`` of bias image 2.
    flat1 : numpy.ndarray
        2D ``numpy.ndarray`` of flat image 1.
    flat2 : array_like
        2D ``numpy.ndarray`` of flat image 2.

    Returns
    -------
    gain : float
        Gain of the camera in electrons/ADU.
    readnoise : float
        Readout noise of the camera in electrons.

    See Also
    --------
    spe_to_dict, gain_readnoise_from_master

    Notes
    -----
    SEQUENCE_NUMBER = 1.2
    from [1]_:
        (b1, b2) = (bias1_mean, bias2_mean)
        diff_b12 = b1 - b2
        sig_db12 = stddev(diff_b12)
        (f1, f2) = (flat1_mean, flat2_mean)
        diff_f12 = f1 - f2
        sig_df12 = stddev(diff_f12)
        gain = ((f1 + f2) - (b1 + b2)) / (sig_df12**2 - sig_db12**2)
        readnoise = gain * sig_db12 / sqrt(2)
    from [2]_:
        Using the median as estimator of average because robust to outliers.
        Using sigmaG as estimator of standard deviation because robust to outliers.
    from [3]_:
        The distribution of the difference of two normally distributed variables is the normal difference distribution
            with sigma12**2 = sigma1**2 + sigma2**2

    References
    ----------
    .. [1] Howell, 2006, "Handbook of CCD Astronomy", sec 4.3 "Calculation of read noise and gain"
    .. [2] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
        sec 3.2, "Descriptive Statistics"
    .. [3] http://mathworld.wolfram.com/NormalDifferenceDistribution.html

    """
    # TODO: As of 2014-08-18, gain_readnoise_from_master and gain_readnoise_from_random do not agree. Check formulas.
    # Note: As of 2014-08-18, `ccdproc.CCDData` objects give `numpy.ndarrays` that with 16-bit unsigned ints.
    # Subtracting these arrays gives values close to 0 and close to 65535. Add variance to circumvent unsigned ints.
    b1 = np.nanmedian(bias1)
    b2 = np.nanmedian(bias2)
    sigmaG_diff_b12 = math.sqrt(astroML_stats.sigmaG(bias1) ** 2.0 + astroML_stats.sigmaG(bias2) ** 2.0)
    f1 = np.nanmedian(flat1)
    f2 = np.nanmedian(flat2)
    sigmaG_diff_f12 = math.sqrt(astroML_stats.sigmaG(flat1) ** 2.0 + astroML_stats.sigmaG(flat2) ** 2.0)
    gain = (((f1 + f2) - (b1 + b2)) / (sigmaG_diff_f12 ** 2.0 - sigmaG_diff_b12 ** 2.0))
    readnoise = gain * sigmaG_diff_b12 / math.sqrt(2.0)
    return (gain * (astropy.units.electron / astropy.units.adu), readnoise * astropy.units.electron) #@UndefinedVariable


# TODO: Once gain_readnoise_from_masters and gain_readnoise_from_random agree, fix and use check_gain_readnoise
# def check_gain_readnoise(bias_dobj, flat_dobj, bias_master = None, flat_master = None,
#                          max_iters=30, max_successes=3, tol_gain=0.01, tol_readnoise = 0.1):
#     """Calculate gain and readnoise using both master images and random images.
#     Compare with image difference/sum method also from sec 4.3. Calculation of read noise and gain, Howell
#     Needed by cosmic ray cleaner.
#     """
#     # Define success criteria for iterations.
#     def success_crit(gain_master, gain_new, gain_old, tol_acc_gain, tol_pre_gain,
#                      readnoise_master, readnoise_new, readnoise_old, tol_acc_readnoise, tol_pre_readnoise):
#         sc = ((abs(gain_new - gain_master) < tol_acc_gain) and
#               (abs(gain_new - gain_old)    < tol_pre_gain) and
#               (abs(readnoise_new - readnoise_master) < tol_acc_readnoise) and
#               (abs(readnoise_new - readnoise_old)    < tol_pre_readnoise))
#         return sc
#     # randomly select 2 bias images and 2 flat images
#     # Accuracy and precision are set to same.
#     # tol_readnoise in electrons. From differences in ProEM cameras on calibration sheet.
#     # tol_gain in electrons/ADU. From differences in ProEM cameras on calibration sheet.
#     # Initialize
#     np.random.seed(0)
#     is_first_iter = True
#     is_converged = False
#     num_consec_success = 0
#     (gain_finl, readnoise_finl) = (None, None)
#     sc_kwargs = {}
#     (sc_kwargs['tol_acc_gain'], sc_kwargs['tol_pre_gain']) = (tol_gain, tol_gain)
#     (sc_kwargs['tol_acc_readnoise'], sc_kwargs['tol_pre_readnoise']) = (tol_readnoise, tol_readnoise)
#     (sc_kwargs['gain_old'], sc_kwargs['readnoise_old']) = (None, None)
#     # TODO: calc masters from dobjs if None.
#     (sc_kwargs['gain_master'], sc_kwargs['readnoise_master']) = gain_readnoise_from_master(bias_master, flat_master)
#     # TODO: Collect an array of values.
#     # TODO: redo new, old. new is new median. old is old median.
#     for iter in xrange(max_iters):
#         # TODO: Use bootstrap sample
#         (sc_kwargs['gain_new'], sc_kwargs['readnoise_new']) = gain_readnoise_from_random(bias1, bias2, flat1, flat2)
#         if not is_first_iter:
#             if (success_crit(**sc_kwargs)):
#                 num_consec_success += 1
#             else:
#                 num_consec_success = 0
#         if num_consec_success >= max_successes:
#             is_converged = True
#             break
#         # Ready for next iteration.
#         (sc_kwargs['gain_old'], sc_kwargs['readnoise_old']) = (sc_kwargs['gain_new'], sc_kwargs['readnoise_new'])
#         is_first_iter = False
#     # After loop.
#     if is_converged:
#         # todo: give details
#         assert iter+1 > max_successes
#         assert ((abs(gain_new - gain_master) < tol_acc_gain) and
#                 (abs(gain_new - gain_old)    < tol_pre_gain) and
#                 (abs(readnoise_new - readnoise_master) < tol_acc_readnoise) and
#                 (abs(readnoise_new - readnoise_old)    < tol_pre_readnoise))
#         logging.info("Calculations for gain and readnoise converged.")
#         (gain_finl, readnoise_finl) = (gain_master, readnoise_master)
#     else:
#         # todo: assertion error statement
#         assert iter == (max_iters - 1)
#         # todo: warning stderr description.
#         logging.warning("Calculations for gain and readnoise did not converge")
#         (gain_finl, readnoise_finl) = (None, None)
#     return(gain_finl, readnoise_finl)


def get_exptime_prog(spe_footer_xml):
    """Get the programmed exposure time in seconds from the string XML footer of an SPE file.

    Parameters
    ----------
    spe_foooter_xml : string
        ``string`` must be properly formatted XML from a Princeton Instruments SPE v3 file footer [1]_.

    Returns
    -------
    exptime_prog_sec : float
        Programmed exposure time in seconds (i.e. the input exposure time from the observer).

    See Also
    --------
    reduce_ccddata, main

    Notes
    -----
    SEQUENCE_NUMBER : 1.3
    Method uses `bs4.BeautifulSoup` to parse the XML ``string``.
    Converts exposure time to seconds from 'ExposureTime' and 'DelayResolution' XML keywords.

    References
    ----------
    .. [1] Princeton Instruments SPE 3.0 File Format Specification
           ftp://ftp.princetoninstruments.com/Public/Manuals/Princeton%20Instruments/
           SPE%203.0%20File%20Format%20Specification.pdf

    """
    footer_xml = BeautifulSoup(spe_footer_xml, 'xml')
    exptime_prog = float(footer_xml.find(name='ExposureTime').contents[0])
    exptime_prog_res = float(footer_xml.find(name='DelayResolution').contents[0])
    exptime_prog_sec = (exptime_prog / exptime_prog_res)
    return exptime_prog_sec


# noinspection PyUnresolvedReferences
def reduce_ccddata(dobj, dobj_exptime=None,
                   bias=None,
                   dark=None, dark_exptime=None,
                   flat=None, flat_exptime=None):
    """Reduce a dict of object dataframes using the master calibration images for bias, dark, and flat.

    All images must be type `ccdproc.CCDData`. Method will do all reductions possible with given master calibration
    images. Method operates on a ``dict`` in order to minimize the number of pre-reduction operations:
    `dark` - `bias`, `flat` - `bias`, `flat` - `dark`.
    Requires exposure time (seconds) for object dataframes.
    If master dark image is provided, requires exposure time for master dark image.
    If master flat image is provided, requires exposure time for master flat image.

    Parameters
    ----------
    dobj : dict
         ``dict`` with ``ccdproc.CCDData`` values. Non-`ccdproc.CCDData` values are retained as metadata.
    dobj_exptime : {None}, float, optional
         Exposure time of images within `dobj`. All images must have the same exposure time.
         Required if `dark` is provided.
    bias : {None}, ccdproc.CCDData, optional
        Master bias image.
    dark : {None}, ccdproc.CCDData, optional
        Master dark image. Will be scaled to match exposure time for `dobj` images and `flat` image.
    dark_exptime : {None}, float or int, optional
        Exposure time of `dark`. Required if `dark` is provided.
    flat : {None}, ccdproc.CCDData, optional
        Master flat image.
    flat_exptime : {None}, float or int, optional
        Exposure time of `flat`. Required if `flat` is provided.

    Returns
    -------
    dobj_reduced : dict
        `dobj` with ``ccdproc.CCDData`` images reduced. ``dict`` keys with non-`ccdproc.CCDData` values
        are also returned in `dobj_reduced`.

    See Also
    --------
    create_master_calib, remove_cosmic_rays, get_exptime_prog, main, logger
    
    Notes
    -----
    SEQUENCE_NUMBER : 2.0
    As of 2014-08-20, correlated errors in image images are not supported by astropy.
    Sequence of operations (following sec 4.5, "Basic CCD Reduction" [1]_):
    - subtract master bias from master dark
    - subtract master bias from master flat
    - scale and subtract master dark from master flat
    - subtract master bias from each object image
    - scale and subtract master dark from each object image
    - divide each object image by corrected master flat

    References
    ----------
    .. [1] Howell, 2006, "Handbook of CCD Astronomy"
    
    """
    # Check input.
    if bias is not None:
        has_bias = True
    else:
        has_bias = False
    if dark is not None:
        has_dark = True
    else:
        has_dark = False
    if flat is not None:
        has_flat = True
    else:
        has_flat = False
    # If there is a `dark` but no `dobj_exptime` or `dark_exptime`:
    if has_dark:
        if (dobj_exptime is None) or (dark_exptime is None):
            raise IOError("If `dark` is provided, both `dobj_exptime` and `dark_exptime` must also be provided.")
    # If there is a `flat` but no `flat_exptime`:
    if has_flat:
        if flat_exptime is None:
            raise IOError("If `flat` is provided, `flat_exptime` must also be provided.")
    # Silence warnings about correlated errors.
    astropy.nddata.conf.warn_unsupported_correlated = False
    # Note: Modify images in-place to reduce memory overhead.
    # Operations:
    # - subtract master bias from master dark
    # - subtract master bias from master flat
    # - scale and subtract master dark from master flat
    if has_bias:
        if has_dark:
            logger.info("Subtracting master bias from master dark.")
            dark = ccdproc.subtract_bias(ccd=dark, master=bias)
        if has_flat:
            logger.info("Subtracting master bias from master flat.")
            flat = ccdproc.subtract_bias(ccd=flat, master=bias)
    if has_dark and has_flat:
        logger.info("Subtracting master dark from master flat.")
        flat = ccdproc.subtract_dark(ccd=flat, master=dark,
                                     dark_exposure=dark_exptime,
                                     data_exposure=flat_exptime,
                                     scale=True)
    # Operations:
    # - subtract master bias from object image
    # - scale and subtract master dark from object image
    # - divide object image by corrected master flat
    print_progress = define_progress(dobj=dobj)
    logger.info("Reducing object images.")
    logger.info("Subtracting master bias from object images: {tf}".format(tf=has_bias))
    logger.info("Subtracting master dark from object images: {tf}".format(tf=has_dark))
    logger.info("Correcting with master flat for object images: {tf}".format(tf=has_flat))
    sorted_image_keys = sorted([key for key in dobj.keys() if isinstance(dobj[key], ccdproc.CCDData)])
    for key in sorted_image_keys:
        if has_bias:
            logger.debug("Subtracting master bias from object image: {key}".format(key=key))
            dobj[key] = ccdproc.subtract_bias(ccd=dobj[key], master=bias)
        if has_dark:
            logger.debug("Subtracting master dark from object image: {key}".format(key=key))
            dobj[key] = ccdproc.subtract_dark(ccd=dobj[key], master=dark,
                                              dark_exposure=dark_exptime,
                                              data_exposure=dobj_exptime,
                                              scale=True)
        if has_flat:
            logger.debug("Correcting with master flat for object image: {key}".format(key=key))
            dobj[key] = ccdproc.flat_correct(ccd=dobj[key], flat=flat)
        print_progress(key=key)
    return dobj


def remove_cosmic_rays(image, contrast=2.0, cr_threshold=4.5, neighbor_threshold=0.45, gain=0.85, readnoise=6.1,
                       **kwargs):
    """Remove cosmic rays from an image.

    Method uses the `photutils` implementation of the LA-Cosmic algorithm [1]_.

    Parameters
    ----------
    image : array_like
        2D ``numpy.ndarray`` of image.
    contrast : {2.0}, float, optional
        Keyword argument for `photutils.detection.lacosmic` [1]_. Chosen from [1]_, and Fig 4 of [2]_.
    cr_threshold : {4.5}, float, optional
        Keyword argument for `photutils.detection.lacosmic` [1]_. Chosen from test script referenced in [3]_.
    neighbor_threshold : {0.45}, float, optional
        Keyword argument for `photutils.detection.lacosmic` [1]_. Chosen from test script referenced in [3]_.
    gain : {0.85}, float, optional
        Keyword argument for `photutils.detection.lacosmic` [1]_. In electrons/ADU. Default is from typical settings for
            Princeton Instruments ProEM 1024B EMCCD [4]_.
    readnoise : {6.1}, float, optional
        Keyword argument for `photutils.detection.lacosmic` [1]_. In electrons. Default is from typical settings for
            Princeton Instruments ProEM 1024B EMCCD [4]_.
    **kwargs :
        Other keyword arguments for `photutils.detection.lacosmic` [1]_.

    Returns
    -------
    image_cleaned : numpy.ndarray
        `image` cleaned of cosmic rays as ``numpy.ndarray``.
    ray_mask : numpy.ndarray
        ``numpy.ndarray`` of booleans with same dimensions as `image_cleaned`s. Pixels where cosmic rays were removed
        are ``True``.
        
    See Also
    --------
    reduce_ccddata, find_stars, main, logger

    Notes
    -----
    SEQUENCE_NUMBER : 3.0
    Use LA-Cosmic algorithm from `photutils` rather than `ccdproc` or `imageutils` until `ccdproc` issue #130 is
        closed [3]_.
    `photutils.detection.lacosmic` is verbose in stdout and stderr.
    
    References
    ----------
    .. [1] http://photutils.readthedocs.org/en/latest/_modules/photutils/detection/lacosmic.html
    .. [2] van Dokkum, 2001. http://adsabs.harvard.edu/abs/2001PASP..113.1420V
    .. [3] https://github.com/astropy/ccdproc/issues/130
    .. [4] Princeton Instruments Certificate of Performance for ProEM 1024B EMCCDs with Traditional Amplifier,
        1 MHz readout speed, gain setting #3 (highest).
    
    """
    # TODO: Silence `photutils.detection.lacosmic`. Hack: http://stackoverflow.com/questions/14058453 and 19425736
    # TODO: Silence `photutils.detection.lacosmic`. directing stdout, stderr causes lacosmic to hang at end.
    tmp_kwargs = dict(contrast=contrast, cr_threshold=cr_threshold, neighbor_threshold=neighbor_threshold,
                      gain=gain, readnoise=readnoise, **kwargs)
    logger.debug("LA-Cosmic keyword arguments: {tmp_kwargs}".format(tmp_kwargs=tmp_kwargs))
    (image_cleaned, ray_mask) = lacosmic.lacosmic(image, contrast=contrast, cr_threshold=cr_threshold,
                                                  neighbor_threshold=neighbor_threshold, gain=gain, readnoise=readnoise,
                                                  **kwargs)
    return image_cleaned, ray_mask


# noinspection PyPep8Naming
def normalize(array):
    """Normalize an array in a robust way.

    The function flattens an array then normalizes in a way that is insensitive to outliers (i.e. ignore stars on an
    image of the night sky). Following [1]_, the function uses `sigmaG` as a width estimator and uses the median as an
    estimator for the mean.
    
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
    find_stars, logger
    
    Notes
    -----
    SEQUENCE_NUMBER : 3.1
    `array_normd` = (`array` - median(`array`)) / `sigmaG`
    `sigmaG` = 0.7413(q75(`array`) - q50(`array`))
    q50, q75 = 50th, 75th quartiles (q50 == median)

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
          sec 3.2, "Descriptive Statistics"
    
    """
    array_np = np.array(array)
    median = np.nanmedian(array_np)
    sigmaG = astroML_stats.sigmaG(array_np)
    if sigmaG == 0:
        logger.warning("SigmaG = 0. Normalized array will be all numpy.NaN")
    array_normd = (array_np - median) / sigmaG
    return array_normd


# noinspection PyUnresolvedReferences
# TODO: don't shadow max_sigma
def find_stars(image, min_sigma=1, max_sigma=max_sigma, num_sigma=2, threshold=3, **kwargs):
    """Find stars in an image and return as a dataframe.
    
    Function normalizes the image [1]_ then uses Laplacian of Gaussian method [2]_ [3]_ to find star-like blobs.
    Method can also find extended sources by modifying `blobargs`, however this pipeline is tailored for stars.

    Parameters
    ----------
    image : array_like
        2D ``numpy.ndarray`` of image.
    min_sigma : {1}, int, optional
        Keyword argument for `skimage.feature.blob_log` [3]_. Smallest sigma (pixels) to use for Gaussian kernel.
    max_sigma : {5}, int, optional
        Keyword argument for `skimage.feature.blob_log` [3]_. Largest sigma (pixels) to use for Gaussian kernel.
    num_sigma : {2}, int, optional
        Keyword argument for `skimage.feature.blob_log` [3]_. Number sigma between smallest and largest sigmas (pixels)
        to use for Gaussian kernel.
    threshold : {3}, int, optional
        Keyword argument for `skimage.feature.blob_log` [3]_. Because image is normalized, `threshold` is the number
        of standard deviations above the image median for counts per source pixel.
    **kwargs:
        Other keyword arguments for `skimage.feature.blob_log` [3]_.

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
    remove_cosmic_rays, center_stars, normalize, make_timestamps_timeseries, max_sigma, plot_stars

    Notes
    -----
    SEQUENCE_NUMBER : 4.0
    Can generalize to extended sources but for increased execution time.
    Use `find_stars` after removing cosmic rays to prevent spurious sources.
    If focus is poor or if PSF is oversampled (FWHM is many pixels), method may find multiple small stars within a
        single star. Use `center_stars` then `combine_stars` to resolve degeneracy in coordinates.

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.2, "Descriptive Statistics"
    .. [2] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    .. [3] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    """
    # Normalize and smooth image then find stars. Order by x,y,sigma.
    image = normalize(image)
    image = scipy.signal.medfilt2d(image, kernel_size=3) #@UndefinedVariable
    stars_arr = feature.blob_log(image=image, min_sigma=min_sigma, max_sigma=max_sigma,
                                 num_sigma=num_sigma, threshold=threshold, **kwargs)
    if len(stars_arr) > 0:
        stars = pd.DataFrame(stars_arr, columns=['y_pix', 'x_pix', 'sigma_pix'])
    else:
        logger.debug("No stars found. num_stars = {num}".format(num=len(stars_arr)))
        stars = pd.DataFrame(columns=['y_pix', 'x_pix', 'sigma_pix'])
    return stars[['x_pix', 'y_pix', 'sigma_pix']]


def plot_stars(image, stars, zoom=None, radius=5, interpolation='none', **kwargs):
    """Plot detected stars overlayed on image.

    Overlay circles around stars and label.
    
    Parameters
    ----------
    image : array_like
        2D ``numpy.ndarray`` of image.
    stars : pandas.DataFrame
        `pandas.DataFrame` with:
        Rows:
            `idx` : 1 index label for each star.
        Columns:
            `x_pix` : x-coordinate (pixels) of star.
            `y_pix` : y-coordinate (pixels) of star.
    zoom : {None}, optional, tuple
        x-y ranges of image for zoom. Must have format ((xmin, xmax), (ymin, ymax)).
        Returned `numpy` image slice is image[ymin: ymax, xmin: xmax].
    radius : {5}, optional, float or int
        The radius of the circle around each star in pixels.
    interpolation : 'none', string, optional
        Keyword argument for `matplotlib.pyplot.imshow`.
    **kwargs :
        Other keyword arguments for `matplotlib.pyplot.imshow`.

    Returns
    -------
    None : None
        Displays a plot with labeled stars using `matplotlib.pyplot.imshow`.

    See Also
    --------
    find_stars, main, make_timestamps_timeseries, plot_positions, plot_lightcurve, plot_matches

    Notes
    -----
    SEQUENCE_NUMBER : 4.1

    Examples
    --------
    ```
    key = 510
    ftnum = object_ccddata[key].meta['frame_tracking_number']
    image = object_ccddata[key].data
    stars = timeseries.loc[ftnum].unstack('quantity_unit')
    print("ftnum = {num}".format(num=ftnum))
    print(stars)
    utils.plot_stars(image=image, stars=stars, zoom=((0, 100), (0, 100)))
    ```

    References
    ----------
    .. [1] http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
    .. [2] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    """
    # If zoom is used, check input dimensions.
    if zoom is None:
        plt.imshow(image, interpolation=interpolation, **kwargs)
        (x_offset, y_offset) = (0, 0)
    else:
        # Check input
        if not (len(zoom) == 2 and len(zoom[0]) == 2 and len(zoom[1]) == 2):
            raise IOError(("Required: Format of `zoom` must be ((xmin, xmax), (ymin, ymax)).\n" +
                           "zoom = {zoom}").format(zoom=zoom))
        ((xmin, xmax), (ymin, ymax)) = zoom
        if not (xmin >= 0 and ymin >= 0):
            raise IOError(("`zoom` = ((xmin, xmax), (ymin, ymax)). Required: xmin and ymin >= 0.\n" +
                           "zoom = {zoom}").format(zoom=zoom))
        if not ((xmax - xmin) >= 1 and (ymax - ymin) >= 1):
            raise IOError(("`zoom` = ((xmin, xmax), (ymin, ymax)). Required: (xmax - xmin) and (ymax - ymin) >= 1.\n" +
                           "zoom = {zoom}").format(zoom=zoom))
        (ydim, xdim) = image.shape
        if not (xmin <= (xdim - 1) and ymin <= (ydim - 1)):
            raise IOError(("`zoom` = ((xmin, xmax), (ymin, ymax)). Required: xmin <= xdim - 1 and ymin <= ydim - 1.\n" +
                           "zoom = {zoom}\n" +
                           "(xdim, ydim) = {xydims}").format(zoom=zoom, xydims=(xdim, ydim)))
        plt.imshow(image[ymin: ymax, xmin: xmax], interpolation=interpolation, **kwargs)
        (x_offset, y_offset) = (xmin, ymin)
    if not (x_offset >= 0 and y_offset >= 0):
        raise AssertionError(("Program error. Required: x_offset and y_offset >= 0\n" +
                              "(x_offset, y_offset) = {xyoffsets}").format(xyoffsets=(x_offset, y_offset)))
    for (idx, x_pix, y_pix) in stars[['x_pix', 'y_pix']].itertuples():
        circle = plt.Circle((x_pix - x_offset, y_pix - y_offset), radius=radius,
                            color='lime', linewidth=2, fill=False)
        plt.gca().add_patch(circle)
        plt.annotate(str(idx), xy=(x_pix - x_offset, y_pix - y_offset), xycoords='data', xytext=(0, 0),
                     textcoords='offset points', color='lime', fontsize=18, rotation=0)
    plt.show()
    return None


def is_odd(num):
    """Determine if a number is equivalent to an odd integer.

    Parameters
    ----------
    num : float

    Returns
    -------
    tf_odd : bool

    See Also
    --------
    center_stars, get_square_subimage
    
    Notes
    -----
    SEQUENCE_NUMBER : 4.2.0
    Uses `math.fabs`, `math.fmod` rather than abs, % [1]_.
    Allows negative numbers.
    (1 - (1E-13)) evaluates as odd.
    (1 + 0j) raises TypeError.

    References
    ----------
    .. [1] https://docs.python.org/2/library/math.html

    """
    # `==` works for floats and ints.
    tf_odd = (math.fabs(math.fmod(num, 2)) == 1)
    return tf_odd


def get_square_subimage(image, position, width=11):
    """Extract a square subimage centered on a coordinate position.

    If the coordinate position is too close to a image edge, a rectangular subimage is returned.

    Parameters
    ----------
    image : numpy.ndarray
        2D ``numpy.ndarray`` of image.
    position : tuple
        (x, y) pixel coordinate position of center of square subimage. Accepts ``float`` or ``int``.
    width : {11}, optional
        `width` x `width` are the dimensions for a square subimage in pixels. Accepts ``float`` or ``int``.
        `width` will be be corrected to be odd and >= 3 so that the subimage is centered on a pixel.

    Returns
    -------
    subimage : numpy.ndarray
        Square subimage with odd number of pixels per side. If the coordinate position is too close to a image edge,
        a rectangular subimage is returned.

    See Also
    --------
    find_stars, subtract_subimage_background, center_stars, logger

    Notes
    -----
    SEQUENCE_NUMBER : 4.2.1
    Uses imageutils.extract_array_2d to extract the subimage [1]_.

    References
    ----------
    .. [1] http://imageutils.readthedocs.org/en/latest/api/imageutils.extract_array_2d.html#imageutils.extract_array_2d

    """
    # Note:
    # - Dimensions of subimage must be odd so that star is centered.
    # - Shape order (width, height) is reverse of position order (x, y).
    # - numpy.ndarrays are ordered by row_idx (y) then col_idx (x). (0,0) is in upper left.
    # - subimage may not be square due to star's proximity to image edge.
    # noinspection PyUnresolvedReferences
    width = np.rint(width)
    if width < 3:
        width = 3
    if not is_odd(width):
        width += 1
    height = width
    subimage = imageutils.extract_array_2d(array_large=image,
                                           shape=(height, width),
                                           position=position)
    (height_actl, width_actl) = subimage.shape
    if (width_actl != width) or (height_actl != height):
        logger.debug(("Star was too close to the edge of the image to extract a square subimage. " +
                      "width={wid}, position={pos}").format(wid=width, pos=position))
    return subimage


# TODO: function obsolete when median background subtracted before centering stars. remove when implemented in make_timestamps_timeseries
# noinspection PyPep8Naming
def subtract_subimage_background(subimage, threshold_sigma=3):
    """Subtract the background intensity from a subimage centered on a source.

    The function estimates the background as the median intensity of pixels bordering the subimage (i.e. square
    aperture photometry). Background sigma is also computed from the border pixels. The median + number of
    selected sigma is subtracted from the subimage. Pixels whose original intensity was less than the
    median + sigma are set to 0.

    Parameters
    ----------
    subimage : array_like
        2D ``numpy.ndarray`` of subimage.
    threshold_sigma : {3}, float or int, optional
        `threshold_sigma` is the number of standard
        deviations above the subimage median for counts per pixel. Pixels with
        fewer counts are set to 0. Uses `sigmaG` [2]_.

    Returns
    -------
    subimage_sub : numpy.ndarray
        Background-subtracted `subimage` as ``numpy.ndarray``.

    See Also
    --------
    get_square_subimage, center_stars

    Notes
    -----
    SEQUENCE_NUMBER : 4.2.2
    The source must be centered to within ~ +/- 1/4 of the subimage width.
    At least 3 times as many border pixels used in estimating the background
        as compared to the source [1]_.
    `sigmaG` = 0.7413(q75(`subimage`) - q50(`subimage`))
    q50, q75 = 50th, 75th quartiles (q50 == median)

    References
    ----------
    .. [1] Howell, 2006, "Handbook of CCD Astronomy", sec 5.1.2, "Estimation of Background"
    .. [2] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.2, "Descriptive Statistics"
        
    """
    subimage_np = np.array(subimage)
    (height, width) = subimage_np.shape
    if width != height:
        raise IOError(("Subimage must be square.\n" +
                       "  width = {wid}\n" +
                       "  height = {ht}").format(wid=width,
                                                 ht=height))
    # Choose border width such ratio of number of background pixels to source pixels is >= 3.
    border = int(math.ceil(width / 4.0))
    arr_longtop_longbottom = np.append(subimage_np[:border],
                                       subimage_np[-border:])
    arr_shortleft_shortright = np.append(subimage_np[border:-border, :border],
                                         subimage_np[border:-border, -border:])
    arr_background = np.append(arr_longtop_longbottom,
                               arr_shortleft_shortright)
    arr_source = subimage_np[border:-border, border:-border]
    if (arr_background.size / arr_source.size) < 3:
        # Howell, 2006, "Handbook of CCD Astronomy", sec 5.1.2, "Estimation of Background"
        raise AssertionError(("Program error. There must be at least 3 times as many sky pixels\n" +
                              "  as source pixels to accurately estimate the sky background level.\n" +
                              "  arr_background.size = {nb}\n" +
                              "  arr_source.size = {ns}").format(nb=arr_background.size,
                                                                 ns=arr_source.size))
    median = np.nanmedian(arr_background)
    sigmaG = astroML_stats.sigmaG(arr_background)
    subimage_sub = subimage_np - (median + threshold_sigma * sigmaG)
    subimage_sub[subimage_sub < 0.0] = 0.0
    return subimage_sub


# noinspection PyUnresolvedReferences
def center_stars(image, stars, box_pix=21, threshold_sigma=3, method='fit_2dgaussian'):
    """Compute centroids of pre-identified stars in an image and return as a dataframe.

    Extract a square subimage around each star. Side-length of the subimage box is `box_pix`.
    With the given method, return a dataframe with sub-pixel coordinates of the centroid and sigma standard deviation.
    Uses a constant `box_pix`, so assumes all stars in the image have the same PSF. This assumption is invalid
    for galaxies.

    Parameters
    ----------
    image : array_like
        2D ``numpy.ndarray`` of image.
    stars : pandas.DataFrame
        ``pandas.DataFrame`` with:
        Rows:
            `idx` : 1 index label for each star.
        Columns:
            `x_pix` : x-coordinate (pixels) of star.
            `y_pix` : y-coordinate (pixels) of star.
            `sigma_pix` : Standard deviation (pixels) of a rough 2D Gaussian fit to the star (usually 1 pixel).
    box_pix : {21}, optional
        `box_pix` x `box_pix` are the dimensions for a square subimage around the source.
        `box_pix` will be corrected to be odd and >= 3 so that the center pixel of the subimage is
        the initial `x_pix`, `y_pix`. Fitting methods converge to within agreement for `box_pix`>=11.
        Typical observed stars fit in `box_pix` = 21.
    threshold_sigma : {3}, optional
        `threshold_sigma` is the number of standard deviations above the subimage median for counts per pixel.
        Accepts ``float`` or ``int``. Pixels with fewer counts are set to 0. Uses `sigmaG` [3]_.
    method : {fit_2dgaussian, fit_bivariate_normal}, optional
        The method by which to compute the centroids and sigma.
        `fit_2dgaussian` : Method is from photutils [1]_ and astropy [2]_. Return the centroid coordinates and
            standard devaition sigma from fitting a 2D Gaussian to the intensity distribution. `fit_2dgaussian`
            executes quickly, agrees with `fit_bivariate_normal`, and converges within agreement
            by `box_pix` = 11. See example below.
        `fit_bivariate_normal` : Model the photon counts within each pixel of the subimage as from a uniform
            distribution [3]_. Return the centroid coordinates and standard deviation sigma from fitting
            a bivariate normal (Gaussian) distribution to the modeled the photon count distribution [4]_.
            `fit_bivariate_sigma` is statistically robust and converges by `box_pix`= 11, but it executes slowly.
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
    find_stars, get_square_subimage, subtract_subimage_background, logger, make_timestamps_timeseries
            
    Notes
    -----
    SEQUENCE_NUMBER : 5.0
    Example: Fitting methods `fit_2dgaussian` and `fit_bivariate_normal` were tested on a bright star with peak
        18000 ADU above background, FHWM ~3.8 pix, initial `sigma_pix` = 1, `box_pix` = 3 to 33. 2014-08-11, STH.
        For `fit_2dgaussian`:
        - For varying subimages, position converges to within 0.01 pix of final solution at 11x11 subimage.
        - For varying subimages, sigma converges to within 0.05 pix of final solution at 11x11 subimage.
        - Final position solution agrees with `fit_bivariate_normal` final position solution within +/- 0.1 pix.
        - Final sigma solution agrees with `fit_bivariate_normal` final sigma solution within +/- 0.2 pix.
        - For 11x11 subimage, method takes ~25 ms. Method scales \propto box_pix.
        For `fit_bivariate_normal`:
        - For varying subimages, position converges to within 0.02 pix of final solution at 11x11 subimage.
        - For varying subimages, sigma converges to within 0.1 pix of final solution at 11x11 subimage.
        - Final position solution agrees with `fit_2dgaussian` final position solution within +/- 0.1 pix.
        - Final sigma solution agrees with `fit_2dgaussian` final sigma solution within +/- 0.2 pix.
        - For 11x11 subimage, method takes ~450 ms. Method scales \propto box_pix**2.
            
    References
    ----------
    .. [1] http://photutils.readthedocs.org/en/latest/photutils/morphology.html#centroiding-an-object
    .. [2] http://astropy.readthedocs.org/en/latest/api/astropy.modeling.functional_models.Gaussian2D.html
    .. [3] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy",
           sec 3.3.1., "The Uniform Distribution"
    .. [4] http://www.astroml.org/book_figures/chapter3/fig_robust_pca.html
    
    """
    # TODO: rewrite to remove subimaging. operate on median-subtracted image.
    # Check input.
    valid_methods = ['fit_2dgaussian', 'fit_bivariate_normal']
    if method not in valid_methods:
        raise IOError(("Invalid method: {meth}\n" +
                       "Valid methods: {vmeth}").format(meth=method, vmeth=valid_methods))
    stars.dropna(subset=['x_pix', 'y_pix'], inplace=True)
    # Make square subimages and compute centroids and sigma by chosen method.
    # Each star or extended source may have a different sigma. Store results in a dataframe.
    stars_init = stars.copy()
    stars_finl = stars.copy()
    stars_finl[['x_pix', 'y_pix', 'sigma_pix']] = np.NaN
    # noinspection PyTypeChecker
    if len(stars) > 0:
        width = int(math.ceil(box_pix))
        for (idx, x_init, y_init, sigma_init) in stars_init[['x_pix', 'y_pix', 'sigma_pix']].itertuples():
            subimage = get_square_subimage(image=image, position=(x_init, y_init), width=width)
            # If the star was too close to the image edge to extract the square subimage, skip the star.
            # Otherwise, compute the initial position for the star relative to the subimage.
            # The initial position relative to the subimage is an integer pixel.
            (height_actl, width_actl) = subimage.shape
            if (width_actl != width) or (height_actl != width):
                logger.debug(("Star was too close to the edge of the image to extract a square subimage. Skipping star. " +
                              "Program variables: {vars}").format(vars={'idx': idx, 'x_init': x_init, 'y_init': y_init,
                                                                        'sigma_init': sigma_init, 'box_pix': box_pix,
                                                                        'width': width, 'width_actl': width_actl,
                                                                        'height_actl': height_actl}))
                continue
            x_init_sub = (width_actl - 1) / 2
            y_init_sub = (height_actl - 1) / 2
            # Compute the centroid position and standard deviation sigma for the star relative to the subimage.
            # using the selected method. Subtract background to fit counts only belonging to the source.
            subimage = subtract_subimage_background(subimage, threshold_sigma)
            # TODO: move methods to own functions.
            if method == 'fit_2dgaussian':
                # Test results: 2014-08-11, STH
                # - Test on star with peak 18k ADU counts above background; FWHM ~3.8 pix.
                # - For varying subimages, position converges to within 0.01 pix of final solution at 11x11 subimage.
                # - For varying subimages, sigma converges to within 0.05 pix of final solution at 11x11 subimage.
                # - Final position solution agrees with `fit_bivariate_normal` final position solution within +/- 0.1 pix.
                # - Final sigma solution agrees with `fit_bivariate_normal` final sigma solution within +/- 0.2 pix.
                # - For 11x11 subimage, method takes ~25 ms. Method scales \propto box_pix.
                # Method description:
                # - See photutils [1]_ and astropy [2]_.
                # - To calculate the standard deviation for the 2D Gaussian:
                # zvec = xvec + yvec
                # xvec, yvec made orthogonal after PCA ('x', 'y' no longer means x,y pixel coordinates)
                # ==> |zvec| = |xvec + yvec| = |xvec| + |yvec|
                # Notation: x = |xvec|, y = |yvec|, z = |zvec|
                # ==> Var(z) = Var(x + y)
                # = Var(x) + Var(y) + 2*Cov(x, y)
                # = Var(x) + Var(y) since Cov(x, y) = 0 due to orthogonality.
                # ==> sigma(z) = sqrt(sigma_x**2 + sigma_y**2)
                fit = morphology.fit_2dgaussian(subimage)
                (x_finl_sub, y_finl_sub) = (fit.x_mean, fit.y_mean)
                sigma_finl_sub = math.sqrt(fit.x_stddev ** 2.0 + fit.y_stddev ** 2.0)
            elif method == 'fit_bivariate_normal':
                # Test results: 2014-08-11, STH
                # - Test on star with peak 18k ADU counts above background; FWHM ~3.8 pix.
                # - For varying subimages, position converges to within 0.02 pix of final solution at 11x11 subimage.
                # - For varying subimages, sigma converges to within 0.1 pix of final solution at 11x11 subimage.
                # - Final position solution agrees with `fit_2dgaussian` final position solution within +/- 0.1 pix.
                # - Final sigma solution agrees with `fit_2dgaussian` final sigma solution within +/- 0.2 pix.
                # - For 11x11 subimage, method takes ~450 ms. Method scales \propto box_pix**2.
                # Method description:
                # - Model the photons hitting the pixels of the subimage and
                # robustly fit a bivariate normal distribution.
                # - Conservatively assume that photons hit each pixel, even those of the star,
                # with a uniform distribution. See [3]_, [4]_.
                # - Seed the random number generator only once per call to this method for reproducibility.
                # - To calculate the standard deviation for the 2D Gaussian:
                # zvec = xvec + yvec
                # xvec, yvec made orthogonal after PCA ('x', 'y' no longer means x,y pixel coordinates)
                # ==> |zvec| = |xvec + yvec| = |xvec| + |yvec|
                # Notation: x = |xvec|, y = |yvec|, z = |zvec|
                # ==> Var(z) = Var(x + y)
                # = Var(x) + Var(y) + 2*Cov(x, y)
                # = Var(x) + Var(y)
                # since Cov(x, y) = 0 due to orthogonality.
                #   ==> sigma(z) = sqrt(sigma_x**2 + sigma_y**2)
                x_dist = []
                y_dist = []
                (height_actl, width_actl) = subimage.shape
                np.random.seed(0)
                for y_idx in xrange(height_actl):
                    for x_idx in xrange(width_actl):
                        pixel_counts = np.rint(subimage[y_idx, x_idx])
                        x_dist_pix = scipy.stats.uniform(x_idx - 0.5, 1) #@UndefinedVariable
                        x_dist.extend(x_dist_pix.rvs(pixel_counts))
                        y_dist_pix = scipy.stats.uniform(y_idx - 0.5, 1) #@UndefinedVariable
                        y_dist.extend(y_dist_pix.rvs(pixel_counts))
                (mu, sigma1, sigma2, alpha) = astroML_stats.fit_bivariate_normal(x_dist, y_dist, robust=True)
                (x_finl_sub, y_finl_sub) = mu
                sigma_finl_sub = math.sqrt(sigma1 ** 2.0 + sigma2 ** 2.0)
            # # NOTE: 2014-08-10, STH
            # # The following methods have been commented out because they do not provide an estimate for the star's
            # # standard deviation as a 2D Gaussian.
            # # elif method == 'centroid_com':
            # # `centroid_com` : Method is from photutils [1]_. Return the centroid from computing the image moments.
            # # Method is very fast but only accurate between 7 <= `box_pix` <= 11 given `sigma`=1 due to
            # # sensitivity to outliers.
            # # Test results: 2014-08-09, STH
            # # - Test on star with peak 18k ADU counts above background; platescale = 0.36 arcsec/superpix;
            # #   seeing = 1.4 arcsec.
            # # - For varying subimages, method does not converge to final centroid solution.
            # # - For 7x7 to 11x11 subimages, centroid solution agrees with centroid_2dg centroid solution within
            # #   +/- 0.01 pix, but then diverges from solution with larger subimages.
            # #   Method is susceptible to outliers.
            #     # - For 7x7 subimages, method takes ~3 ms per subimage. Method is invariant to box_pix and always
            #     #   takes ~3 ms.
            #     # (x_finl_sub, y_finl_sub) = morphology.centroid_com(subimage)
            #     # elif method == 'fit_max_phot_flux':
            #     # `fit_max_phot_flux` : Method is from Mike Montgomery, UT Austin, 2014. Return the centroid from
            #     # computing the centroid that yields the largest photometric flux. Method is fast, but,
            #     # as of 2014-08-08 (STH), implementation is inaccurate by ~0.1 pix (given `sigma`=1, `box_pix`=7),
            #     # and method is possibly sensitive to outliers.
            #     # Test results: 2014-08-09, STH
            #     # - Test on star with peak 18k ADU counts above background; platescale = 0.36 arcsec/superpix;
            #     #   seeing = 1.4 arcsec.
            #     # - For varying subimages, method converges to within +/- 0.0001 pix of final centroid solution at
            #     #   7x7 subimage, however final centroid solution disagrees with other methods' centroid solutions.
            #     # - For 7x7 subimage, centroid solution disagrees with centroid_2dg centroid solution for 7x7 subimage
            #     #   by ~0.1 pix. Method may be susceptible to outliers.
            #     # - For 7x7 subimage, method takes ~130 ms. Method scales \propto box_pix.
            #     # TODO: Test different minimization methods
            #     def obj_func(subimage, position, radius):
            #         """Objective function to minimize: -1*photometric flux from star.
            #         Assumed to follow a 2D Gaussian point-spread function.
            #
            #         Parameters
            #         ----------
            #         subimage : array_like
            #             2D subimage of image. Used only by `obj_func`.
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
            #         .. [1] http://photutils.readthedocs.org/en/latest/api/
            #                photutils.CircularAperture.html#photutils.CircularAperture
            #         .. [2] http://photutils.readthedocs.org/en/latest/api/
            #                photutils.aperture_photometry.html#photutils.aperture_photometry
            #         .. [3] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html
            #
            #         """
            #         aperture = ('circular', radius)
            #         (flux_table, aux_dict) = photutils.aperture_photometry(subimage, position, aperture)
            #         flux_neg = -1. * flux_table['aperture_sum'].data
            #         return flux_neg
            #
            #     def jac_func(subimage, position, radius, eps=0.005):
            #         """Jacobian of the objective function for fixed radius.
            #         Assumed to follow a 2D Gaussian point-spread function.
            #
            #         Parameters
            #         ----------
            #         subimage : array_like
            #             2D subimage of image. Used only by `obj_func`
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
            #         .. [1] http://photutils.readthedocs.org/en/latest/api/
            #                photutils.CircularAperture.html#photutils.CircularAperture
            #         .. [2] http://photutils.readthedocs.org/en/latest/api/
            #                photutils.aperture_photometry.html#photutils.aperture_photometry
            #         .. [3] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html
            #
            #         """
            #         try:
            #             [x_pix, y_pix] = position
            #         except ValueError:
            #             raise ValueError(("'position' must have the format [x_pix, y_pix]\n"+
            #                               "  position = {pos}").format(pos=position))
            #         jac = np.zeros(len(position))
            #         fxp1 = obj_func(subimage, (x_pix + eps, y_pix), radius)
            #         fxm1 = obj_func(subimage, (x_pix - eps, y_pix), radius)
            #         fyp1 = obj_func(subimage, (x_pix, y_pix + eps), radius)
            #         fym1 = obj_func(subimage, (x_pix, y_pix - eps), radius)
            #         jac[0] = (fxp1-fxm1)/(2.*eps)
            #         jac[1] = (fyp1-fym1)/(2.*eps)
            #         return jac
            #
            #     position = [x_init_sub, y_init_sub]
            #     radius = sigma_to_fwhm(sigma_init)
            #     res = scipy.optimize.minimize(fun=(lambda pos: obj_func(subimage, pos, radius)),
            #                                   x0=position,
            #                                   method='L-BFGS-B',
            #                                   jac=(lambda pos: jac_func(subimage, pos, radius)),
            #                                   bounds=((0, width), (0, height)))
            #     (x_finl_sub, y_finl_sub) = res.x
            else:
                    raise AssertionError("Program error. Method not accounted for:\n{meth}".format(meth=method))
            # Compute the centroid coordinates relative to the entire image.
            # Return the dataframe with centroid coordinates and sigma.
            (x_offset, y_offset) = (x_finl_sub - x_init_sub,
                                    y_finl_sub - y_init_sub)
            (x_finl, y_finl) = (x_init + x_offset,
                                y_init + y_offset)
            sigma_finl = sigma_finl_sub
            stars_finl.loc[idx, ['x_pix', 'y_pix', 'sigma_pix']] = (x_finl, y_finl, sigma_finl)
    else:
        # noinspection PyTypeChecker
        logger.debug("No stars to center. num_stars = {num}".format(num=len(stars)))
    return stars_finl


def drop_duplicate_stars(stars):
    """Stars within 2 sigma of each other are assumed to be the same star.

    Parameters
    ----------
    stars : pandas.DataFrame

    Returns
    -------
    stars : pandas.DataFrame

    See Also
    --------
    logger, center_stars, make_timestamps_timeseries

    Notes
    -----
    SEQUENCE_NUMBER : 6.0

    """
    # Remove all NaN values and sort `stars` by `sigma_pix` so that sources with larger sigma contain the
    # duplicate sources with smaller sigma.
    # Note: `stars` is dynamically updated at the end of each iteration.
    # noinspection PyUnresolvedReferences
    stars.dropna(subset=['x_pix', 'y_pix'], inplace=True)
    # noinspection PyTypeChecker
    if len(stars) > 1:
        for (idx, row) in stars.sort(columns=['sigma_pix']).iterrows():
            # Check length again since `stars` is dynamically updated at the end of each iteration.
            # noinspection PyTypeChecker
            if len(stars) > 1:
                min_dist = None
                min_idx2 = None
                update_dist = None
                # TODO: vectorize
                for (idx2, row2) in stars.drop(idx, inplace=False).iterrows():
                    dist = scipy.spatial.distance.euclidean(u=row.loc[['x_pix', 'y_pix']], #@UndefinedVariable
                                                            v=row2.loc[['x_pix', 'y_pix']])
                    if min_dist is None:
                        update_dist = True
                    elif dist < min_dist:
                        update_dist = True
                    else:
                        update_dist = False
                    if update_dist:
                        min_dist = dist
                        min_idx2 = idx2
                # Accept stars at least 2*max(sigma_pix) away.
                # Note: Faint stars undersample the PSF given a noisy background and are calculated to have smaller
                # sigma than the actual sigma of the PSF.
                # TODO: calculate psf from image. Use values from psf instead of fixed pixel values?
                if min_dist < 2.0 * np.nanmax([max_sigma, row.loc['sigma_pix'], stars.loc[min_idx2, 'sigma_pix']]):
                    if row.loc['sigma_pix'] >= stars.loc[min_idx2, 'sigma_pix']:
                        raise AssertionError(("Program error. Indices of degenerate stars were not dropped.\n" +
                                              "row:\n{row}\nstars:\n{stars}").format(row=row, stars=stars))
                    logger.debug("Dropping duplicate star:\n{row}".format(row=row))
                    stars.drop(idx, inplace=True)
            else:
                # noinspection PyTypeChecker
                logger.debug("No more duplicate stars to drop. num_stars = {num}".format(num=len(stars)))
    else:
        # noinspection PyTypeChecker
        logger.debug("No duplicate stars to drop. num_stars = {num}".format(num=len(stars)))
    return stars


def translate_images_1to2(image1, image2):
    """Calculate integer-pixel image translation from phase correlation.

    translation = image2_coords - image1_coords so that
    image1_coords + translation = image2_coords
    translation = (`dx_pix`, `dy_pix`)

    Parameters
    ----------
    image1 : numpy.ndarray
    image2 : numpy.ndarray

    Returns
    -------
    dx_pix : float
        `dx_pix` = image2_coord_x - image1_coord_x
    dy_pix : float
        `dy_pix` = image2_coord_y - image1_coord_y

    See Also
    --------
    match_stars, logger

    Notes
    -----
    SEQUENCE_NUMBER : 6.7
    Translation from phase correlation is useful for >0.5 pixel image transfroms.
    Use matched stars for sub-pixel transforms.
    Only use on images with detected stars. Phase correlation is median-filtered, but clouds cause correlation
        to be very noisy.

    References
    ----------
    .. [1] Adapted from http://www.lfd.uci.edu/~gohlke/code/imreg.py.html
    .. [2] http://docs.scipy.org/doc/scipy/reference/generated/
        scipy.interpolate.griddata.html#scipy.interpolate.griddata

    Examples
    --------
    ```
    from __future__ import print_function
    import skimage
    import utils
    stars1 = utils.find_stars(image1)
    print(stars1)
    translation = utils.translate_images_1to2(image1, image2)
    print(translation)
    tform = skimage.transform.SimilarityTransform(translation=translation)
    print(tform(stars1[['x_pix', 'y_pix']].values))
    stars2 = utils.find_stars(image2)
    print(stars2)
    ```

    """
    # Check input.
    if np.all(image1 == image2):
        logger.info("Images are identical.")
        (dx_pix, dy_pix) = (0.0, 0.0)
        return dx_pix, dy_pix
    if image1.shape != image2.shape:
        raise IOError(("Images must have the same shape:\n" +
                       "image1.shape = {s1}\n" +
                       "image2.shape = {s2}").format(s1=image1.shape, s2=image2.shape))
    if image1.ndim != 2:
        raise IOError(("Images must be 2D:\n" +
                       "image1.ndim = {n1}\n" +
                       "image2.ndim = {n2}").format(n1=image1.ndim, n2=image2.ndim))
    # Compute the maximum phase correlation for to determine the image translation.
    # Tile the phase correlation image since the coordinate for maximum phase correlation is usually
    #     near the domain edge. (Tiling allows smoothing the correlation without biasing the max away from the image
    #     boundaries. Tile the image, don't mirror, since the phase correlation is continuous across image boundaries.)
    # Smooth the phase correlation using a median filter since poor focus and clouds will make the correlation noisy.
    # The first estimate for image translation is to an integer pixel.
    # Then use 2D cubic interpolation to determine maximum phase correlation to sub-pixel precision.
    # Note: numpy is row-major: (y_pix, x_pix)
    shape = image1.shape
    f1 = np.fft.fft2(image1)
    f2 = np.fft.fft2(image2)
    phase_corr = abs(np.fft.ifft2((f1.conjugate() * f2) / (abs(f1) * abs(f2))))
    (tiled_offset_y, tiled_offset_x) = shape
    tiled = np.tile(phase_corr, (3, 3))
    tiled = scipy.signal.medfilt2d(tiled, kernel_size=3) #@UndefinedVariable
    # Note: Determine the max phase correlation from the center tile,
    # otherwise max will be in upper right tile and not original image.
    (dy_int, dx_int) = \
        np.unravel_index(
            int(np.nanargmax(tiled[tiled_offset_y : 2*tiled_offset_y, tiled_offset_x : 2*tiled_offset_x])),
            shape)
    (dy_pix, dx_pix) = (dy_int, dx_int)
    # TODO: Do better subpixel translation. Implementation below is off by +/- 0.5 pix in x and y
    #      (due to limit from information theory?) STH, 2014-10-06
    ####################
    # BEGIN SUBPIXEL PHASE CORRELATION
    ####################
    # (tiled_dy_int, tiled_dx_int) = np.add((tiled_offset_y, tiled_offset_x), (dy_int, dx_int))
    # subimg_halfwidth = 5
    # subimg_subpix_res = 0.1
    # subimg = \
    #     tiled[
    #         tiled_dy_int - subimg_halfwidth : tiled_dy_int + subimg_halfwidth + 1,
    #         tiled_dx_int - subimg_halfwidth : tiled_dx_int + subimg_halfwidth + 1]
    # # Note: numpy.ndarray.flatten collapses array so that x index advances faster than y index.
    # subimg_pixels = \
    #     [(x, y)
    #      for y in xrange(tiled_dy_int - subimg_halfwidth, tiled_dy_int + subimg_halfwidth + 1)
    #      for x in xrange(tiled_dx_int - subimg_halfwidth, tiled_dx_int + subimg_halfwidth + 1)]
    # subimg_values = subimg.flatten()
    # (subimg_x_subpix, subimg_y_subpix) = \
    #     np.mgrid[
    #         tiled_dx_int - subimg_halfwidth : tiled_dx_int + subimg_halfwidth + 1 : subimg_subpix_res,
    #         tiled_dy_int - subimg_halfwidth : tiled_dy_int + subimg_halfwidth + 1 : subimg_subpix_res]
    # # Note: scipy.interpolate.griddata uses (x, y) order. numpy uses (y, x) order.
    # subimg_interp = scipy.interpolate.griddata(subimg_pixels, subimg_values,
    #                                            (subimg_x_subpix, subimg_y_subpix), method='cubic')
    # subimg_shape = (len(subimg_y_subpix), len(subimg_x_subpix))
    # (subimg_y_subidx, subimg_x_subidx) = np.unravel_index(int(np.nanargmax(subimg_interp)), subimg_shape)
    # (subimg_dy_pix, subimg_dx_pix) = (subimg_y_subpix[0, subimg_y_subidx], subimg_x_subpix[subimg_x_subidx, 0])
    # (dy_pix, dx_pix) = np.subtract((subimg_dy_pix, subimg_dx_pix), (tiled_offset_y, tiled_offset_x))
    ####################
    # END SUBPIXEL PHASE CORRELATION
    ####################
    # Restrict all coordinates to be relative to first quandrant (containing (1,1))
    # because phase correlation image is continuous across boundaries, .
    if dy_pix > shape[0] / 2.0:
        dy_pix -= shape[0]
    if dx_pix > shape[1] / 2.0:
        dx_pix -= shape[1]
    logger.debug(("Image translation:\n" +
                  "image2_coords - image1_coords = (dx_pix, dy_pix) = {tup}").format(tup=(dx_pix, dy_pix)))
    # Returned order should be (x, y) to be consistent with rest of utils convention.
    return dx_pix, dy_pix


# noinspection PyUnresolvedReferences
def plot_matches(image1, image2, stars1, stars2):
    """Visualize image matching.

    Green line connect matched stars. Red lines connect mismatched stars.

    Parameters
    ----------
    image1 : numpy.ndarray
    image2 : numpy.ndarray
    stars1 : pandas.DataFrame
    stars2 : pandas.DataFrame

    Returns
    -------
    None

    See Also
    --------
    match_stars, logger

    Notes
    -----
    SEQUENCE_NUMBER : 6.8

    References
    ----------
    http://scikit-image.org/docs/dev/auto_examples/plot_matching.html

    """
    # Check input. Reindex so pandas dataframe indices match numpy ndarray indices.
    if (stars1['verif1to2'] != stars2['verif2to1']).any():
        raise IOError(("Dataframe indices are not verified as matching by 'verif1to2', 'verif2to1' columns:\n" +
                       "stars1: {stars1}\n" +
                       "stars2: {stars2}\n").format(stars1=stars1, stars2=stars2))
    (fig, axes) = plt.subplots(nrows=2, ncols=1)
    stars1.reset_index(drop=True, inplace=True)
    stars2.reset_index(drop=True, inplace=True)
    keypoints1 = stars1[['y_pix', 'x_pix']].values
    keypoints2 = stars2[['y_pix', 'x_pix']].values
    # noinspection PyPep8
    matches = stars1[stars1['verif1to2'] == True].index.tolist()
    feature.plot_matches(ax=axes[0], image1=image1, image2=image2, keypoints1=keypoints1, keypoints2=keypoints2,
                         matches=np.column_stack((matches, matches)), keypoints_color='yellow',
                         matches_color='green')
    axes[0].axis('off')
    axes[0].set_title('Verified matches (left: image1, right: image2)')
    # noinspection PyPep8
    not_matches = stars1[stars1['verif1to2'] == False].index.tolist()
    feature.plot_matches(ax=axes[1], image1=image1, image2=image2, keypoints1=keypoints1, keypoints2=keypoints2,
                         matches=np.column_stack((not_matches, not_matches)), keypoints_color='yellow',
                         matches_color='red')
    axes[1].axis('off')
    axes[1].set_title('Unverified matches (left: image1, right: image2)')
    plt.show()
    return None


def match_stars(image1, image2, stars1, stars2, test=False):
    """Match stars.

    Parameters
    ----------
    image1 : numpy.ndarray
    image2 : numpy.ndarray
    stars1 : pandas.DataFrame
    stars2 : pandas.DataFrame
    test : {False, True}, bool, optional
        Shows `plot_matches` if ``True``.

    Returns
    -------
    matched_stars : pandas.DataFrame

    See Also
    --------
    logger, plot_matches, translate_images_1to2, drop_duplicate_stars, make_timestamps_timeseries

    Notes
    -----
    SEQUENCE_NUMBER : 6.9

    """
    # Drop NaNs and check input.
    stars1.dropna(subset=['x_pix', 'y_pix'], inplace=True)
    stars2.dropna(subset=['x_pix', 'y_pix'], inplace=True)
    num_stars1 = len(stars1)
    num_stars2 = len(stars2)
    if num_stars1 < 1:
        raise IOError(("stars1 must have at least one star.\n" +
                       "stars1 = {stars1}").format(stars1=stars1))
    if num_stars1 != num_stars2:
        logger.debug(("`image1` and `image2` have different numbers of stars. There may be clouds.\n" +
                      "num_stars1 = {n1}, num_stars2 = {n2}").format(n1=num_stars1, n2=num_stars2))
    # Create heirarchical dataframe for tracking star matches. Match from star1 positions to star2 positions.
    # `stars` dataframe has the same number of stars as `stars1`.
    # Sort columns to permit heirarchical slicing.
    df_stars1 = stars1.copy()
    df_stars1['verif1to2'] = np.NaN
    df_tform1to2 = (stars1.copy())[['x_pix', 'y_pix']]
    df_tform1to2[:] = np.NaN
    df_stars2 = stars1.copy()
    df_stars2[:] = np.NaN
    df_stars2['idx2'] = np.NaN
    df_stars2['verif2to1'] = np.NaN
    df_stars2['min_dist'] = np.NaN
    df_dict = {'stars1': df_stars1,
               'tform1to2': df_tform1to2,
               'stars2': df_stars2}
    stars = pd.concat(df_dict, axis=1)
    stars.sort_index(axis=1, inplace=True)
    # If any stars exist in stars2, match them...
    if num_stars2 > 0:
        # Use least Euclidean distance to match stars. Verify that matched stars are within 1 sigma of each other
        # and are matched 1-to-1.
        # Note:
        # - Faint stars undersample the PSF given a noisy background and are calculated to have smaller
        #     sigma than the actual sigma of the PSF.
        # - Use .loc with ('tform1to2', ['x_pix', 'y_pix']) to prevent Type Error.
        # TODO: Report pandas bug with Type Error ('tform1to2', ['x_pix', 'y_pix'])?
        # TODO: Allow users to define stars by hand instead of by `find_stars`
        # TODO: Can't individually set pandas.DataFrame elements to True. Report bug?
        translation = translate_images_1to2(image1=image1, image2=image2)
        tform = skimage.transform.SimilarityTransform(translation=translation) #@UndefinedVariable
        stars.loc[:, ('tform1to2', ['x_pix', 'y_pix'])] = tform(stars.loc[:, ('stars1', ['x_pix', 'y_pix'])].values)
        logger.debug("Transform parameters:\n{pars}".format(pars={'translation': tform.translation,
                                                                  'rotation': tform.rotation,
                                                                  'scale': tform.scale,
                                                                  'params': tform.params}))
        # For comparing distances, starsN dataframe with fewest stars is index so that mapping is injective.
        if num_stars1 <= num_stars2:
            stars_dist = pd.DataFrame(index=stars1.index, columns=stars2.index)
            stars_dist.index.names = ['stars1_index']
            stars_dist.columns.names = ['stars2_index']
        else:
            stars_dist = pd.DataFrame(index=stars2.index, columns=stars1.index)
            stars_dist.index.names = ['stars2_index']
            stars_dist.columns.names = ['stars1_index']
        # Make dataframes to cross-check that stars are matched correctly.
        stars1_verified = pd.DataFrame(columns=stars1.columns)
        stars1_unverified = stars1.copy()
        stars2_verified = pd.DataFrame(columns=stars2.columns)
        stars2_unverified = stars2.copy()
        # Compute distances from each translated star coordinate to each found star coordinate.
        # Notation:
        #     idx_r = index row (i.e. if `stars_dist` index is 'stars1_index', idx_r is index from stars1)
        #     idx_c = index column
        for (idx1, row1) in stars.iterrows():
            for (idx2, row2) in stars2.iterrows():
                if stars_dist.index.names[0] == 'stars1_index':
                    (idx_r, idx_c) = (idx1, idx2)
                elif stars_dist.index.names[0] == 'stars2_index':
                    (idx_r, idx_c) = (idx2, idx1)
                else:
                    raise AssertionError(("Program error. `stars_dist` index name should be " +
                                          "'stars1_index' or 'stars2_index':\n" +
                                          "stars_dist =\n{stars_dist}").format(stars_dist=stars_dist))
                stars_dist.loc[idx_r, idx_c] =  \
                    scipy.spatial.distance.euclidean(u=row1.loc['tform1to2', ['x_pix', 'y_pix']], #@UndefinedVariable
                                                     v=row2.loc[['x_pix', 'y_pix']])
        logger.debug("Distances between translated stars from stars1 and newly found stars from stars2.\n" +
                     "stars_dist =\n{sd}".format(sd=stars_dist))
        # Note: Index of stars_dist is translated stars (stars1) or found stars (stars2), whichever is fewer in number.
        # Without loss of generality, assuming index of stars_dist is translated stars:
        #     For every translated star coordinate, match to nearest found star coordinate.
        #     And for every found star coordinate, match to nearest translated star coordinate.
        #     Check that original translated star coordinate is the matched translated star coordinate.
        #     Check that the distance between matched coordinates is within 2*max(sigma_pix) since each coordinate is
        #     only known to +/- sigma..
        #     Keep track of verified stars to prevent over counting.
        # Notation:
        #     idx_r = index row (i.e. if `stars_dist` index is 'stars1_index', idx_r is index from stars1)
        #     idx_rtoc = index mapped from row to column
        for idx_r in stars_dist.index:
            min_idx_rtoc = stars_dist.loc[idx_r].argmin()
            min_dist_rtoc = stars_dist.loc[idx_r].min()
            min_idx_ctor = stars_dist.loc[:, min_idx_rtoc].argmin()
            min_dist_ctor = stars_dist.loc[:, min_idx_rtoc].min()
            # If this star's closest match is unique...
            if (idx_r == min_idx_ctor) and (min_dist_rtoc == min_dist_ctor):
                # Identify index, columns as stars1 or stars2.
                if stars_dist.index.names[0] == 'stars1_index':
                    (idx1, idx2) = (idx_r, min_idx_rtoc)
                elif stars_dist.index.names[0] == 'stars2_index':
                    (idx1, idx2) = (min_idx_rtoc, idx_r)
                else:
                    raise AssertionError(("Program error. `stars_dist` index name should be " +
                                          "'stars1_index' or 'stars2_index':\n" +
                                          "stars_dist =\n{stars_dist}").format(stars_dist=stars_dist))
                # If the nearest star is within 2*max(sigma_pix)...
                verify_match = None
                if min_dist_rtoc < 2.0 * np.nanmax([max_sigma,
                                                    stars.loc[idx1, ('stars1', 'sigma_pix')],
                                                    stars2.loc[idx2, 'sigma_pix']]):
                    verify_match = True
                else:
                    verify_match = False
                # If match was verified, update values and cross-check...
                if verify_match:
                    row1 = stars1.loc[idx1]
                    if idx1 not in stars1_verified.index:
                        stars1_verified.loc[idx1] = row1
                        stars1_unverified.drop(idx1, inplace=True)
                        stars.loc[idx1, ('stars1', 'verif1to2')] = 1
                    else:
                        raise AssertionError(("Program error. Star from stars1 already verified:\n" +
                                              "row from stars1 =\n"
                                              "{row1}").format(row1=row1))
                    row2 = stars2.loc[idx2]
                    if idx2 not in stars2_verified.index:
                        stars2_verified.loc[idx2] = row2
                        stars2_unverified.drop(idx2, inplace=True)
                        stars.loc[idx1, 'stars2'].update(row2)
                        stars.loc[idx1, ('stars2', ['idx2', 'min_dist', 'verif2to1'])] = (idx2, min_dist_rtoc, 1)
                    else:
                        raise AssertionError(("Program error. Star from stars2 already verified:\n" +
                                              "row from stars2 =\n"
                                              "{row2}").format(row2=row2))
                # ...Else match was not verified so use translated coordinates as positions for stars2.
                else:
                    stars.loc[idx1, 'stars2'].update(stars.loc[idx1, 'tform1to2'])
                    stars.loc[idx1, ('stars1', 'verif1to2')] = 0
                    stars.loc[idx1, ('stars2', 'verif2to1')] = 0
                    logger.debug(("Star not verified:\n" +
                                  "row from stars =\n" +
                                  "{row}").format(row=stars.loc[idx1]))
            # ...Else matched star was not the closest and/or not 1-to-1 so use translated coordinates
            # as positions for stars2.
            else:
                stars.loc[idx1, 'stars2'].update(stars.loc[idx1, 'tform1to2'])
                stars.loc[idx1, ('stars1', 'verif1to2')] = 0
                stars.loc[idx1, ('stars2', 'verif2to1')] = 0
                logger.debug(("Star not verified:\n" +
                              "row from stars =\n" +
                              "{row}").format(row=stars.loc[idx1]))
        # Check that all stars have been accounted for. Stars without matches have NaNs in 'star1' or 'star2'.
        # If stars in stars1 were not verified, use translated coordinates as positions in new image. (A transformation
        # from feature detection gives subpixel precision for translation, otherwise star positions will drift over
        # many frames.)
        # If stars in stars2 were not verified, append new stars to positions in new image.
        # TODO: pandas.DataFrame.update doesn't work when using true/false mask. report bug?
        if (len(stars1_verified) != num_stars1) or (len(stars1_unverified) != 0):
            logger.debug(("Not all stars in stars1 were verified as matching stars in stars2:\n" +
                          "stars1_unverified =\n" +
                          "{s1u}").format(s1u=stars1_unverified))
            tfmask_verif = (stars[('stars1', 'verif1to2')] == 1) & (stars[('stars2', 'verif2to1')] == 1)
            translation_subpix = np.nanmean(stars.loc[tfmask_verif, ('stars2', ['x_pix', 'y_pix'])].values -
                                            stars.loc[tfmask_verif, ('stars1', ['x_pix', 'y_pix'])].values,
                                            axis=0)
            tform_subpix = skimage.transform.SimilarityTransform(translation=translation_subpix) #@UndefinedVariable
            tfmask_unverif = -tfmask_verif
            stars.loc[tfmask_unverif, ('tform1to2', ['x_pix', 'y_pix'])] = \
                tform_subpix(stars.loc[tfmask_unverif, ('stars1', ['x_pix', 'y_pix'])].values)
            stars.loc[tfmask_unverif, ('stars2', ['x_pix', 'y_pix'])] = \
                stars.loc[tfmask_unverif, ('tform1to2', ['x_pix', 'y_pix'])].values
            stars.loc[tfmask_unverif, ('stars1', 'verif1to2')] = 0
            stars.loc[tfmask_unverif, ('stars2', 'verif2to1')] = 0
            if stars.loc[:, ('stars2', ['x_pix', 'y_pix'])].isnull().any().any():
                raise AssertionError(("Program error. " +
                                      "('stars2', ['x_pix', 'y_pix']) should not have any null values:\n" +
                                      "stars =\n{stars}").format(stars=stars))
        if (len(stars2_verified) != num_stars2) or (len(stars2_unverified) != 0):
            logger.debug(("Not all stars in stars2 were verified as matching stars in stars1:\n" +
                          "stars2_unverified =\n" +
                          "{s2u}").format(s2u=stars2_unverified))
            df_dict = {'stars1': stars['stars1'],
                       'tform1to2': stars['tform1to2'],
                       'stars2': (stars['stars2']).append(stars2_unverified, ignore_index=True)}
            stars = pd.concat(df_dict, axis=1)
            if stars.loc[:, ('stars2', ['x_pix', 'y_pix'])].isnull().any().any():
                raise AssertionError(("Program error. " +
                                      "('stars2', ['x_pix', 'y_pix']) should not have any null values:\n" +
                                      "stars =\n{stars}").format(stars=stars))
    # ...Else there are no stars in stars2.
    # Assume star coordinates are unchanged.
    else:
        # Use .loc with ('tform1to2', ['x_pix', 'y_pix']) to prevent Type Error.
        # TODO: Report pandas bug with Type Error for ('tform1to2', ['x_pix', 'y_pix'])?
        logger.debug("No stars in stars2. Assuming stars2 (x, y) are same as stars1 (x, y).")
        stars.loc[:, ('tform1to2', ['x_pix', 'y_pix'])] = stars.loc[:, ('stars1', ['x_pix', 'y_pix'])].values
        stars.loc[:, ('stars2', ['x_pix', 'y_pix'])] = stars.loc[:, ('stars1', ['x_pix', 'y_pix'])].values
        stars[('stars1', 'verif1to2')] = 0
        stars[('stars2', 'verif2to1')] = 0
        if stars.loc[:, ('stars2', ['x_pix', 'y_pix'])].isnull().any().any():
            raise AssertionError(("Program error. " +
                                  "('stars2', ['x_pix', 'y_pix']) should not have any null values:\n" +
                                  "stars =\n{stars}").format(stars=stars))
    # Sort columns to permit heirarchical slicing.
    # Replace 1 with True; 0/NaN with False.
    stars.sort_index(axis=1, inplace=True)
    stars[('stars1', 'verif1to2')] = (stars[('stars1', 'verif1to2')] == 1)
    stars[('stars2', 'verif2to1')] = (stars[('stars2', 'verif2to1')] == 1)
    logger.debug("Match stars result:\n{stars}".format(stars=stars))
    # Show plot for testing.
    if test:
        plot_matches(image1=image1, image2=image2, stars1=stars['stars1'], stars2=stars['stars2'])
    # Report results.
    df_dict = {'stars1': stars['stars1'],
               'stars2': stars['stars2'].drop(['idx2', 'min_dist'], axis=1)}
    matched_stars = pd.concat(df_dict, axis=1)
    return matched_stars


def make_timestamps_timeseries(dobj, radii=np.arange(0.5, 10.5, 0.5)):
    """Calculate timeseries lightcurves from data and return tuple: timestamps, timeseries

    Parameters
    ----------
    dobj : dict
        ``dict`` of ``ccdproc.CCDData``
    radii : {numpy.arange(0.5, 10.5, 0.5)}, list, optional
        ``list`` of floats for radii of apertures.
         Default list is `radii` = numpy.arange(0.5, 10.5, 0.5) = [0.5, 1.0, 1.5, ... 9.0, 9.5, 10.0]

    Returns
    -------
    timestamps : pandas.DataFrame
        row names:
            `frame_tracking_number`: Frame tracking number from SPE metadata, >= 1. Example: 1
        column names:
            `quantity` : Timestamps for `exposure_start_timestamp_UTC`, `exposure_mid_timestamp_UTC`,
                `exposure_end_timestamp_UTC` as ``datetime.datetime``
    timeseries : pandas.DataFrame
        row names:
            `frame_tracking_number`: Frame tracking number from SPE metadata, >= 1. Example: 1
        column names:
            `star_index` : Unique index label for stars, >= 0. Example: 0
            `quantity_unit` : Quantities calculated with units, separated by an underscore.
                Examples: `sigma_pix`; `(flux_ADU, 6.0)`, 6.0 is aperture radius in pixels.

    See Also
    --------
    main, logger, match_stars, make_lightcurves

    Notes
    -----
    SEQUENCE_NUMBER : 7.0

    """
    # TODO: let users define own stars
    # TODO: Just build dataframe directly. Don't convert from dict.
    print_progress = define_progress(dobj=dobj)
    sorted_image_keys = sorted([key for key in dobj.keys() if isinstance(dobj[key], ccdproc.CCDData)])
    timestamps_dict = {}
    timeseries_dict = {}
    logger.info("Photometry aperture radii: {radii}".format(radii=radii))
    for key in sorted_image_keys:
        logger.debug("Key: {key}".format(key=key))
        image_new = dobj[key].data
        ftnum_new = dobj[key].meta['frame_tracking_number']
        logger.debug("Frame tracking number: {ftnum}".format(ftnum=ftnum_new))
        stars_new = find_stars(image=image_new)
        logger.debug("Found stars:\n{stars}".format(stars=stars_new))
        stars_new = center_stars(image=image_new, stars=stars_new)
        logger.debug("Centered stars:\n{stars}".format(stars=stars_new))
        stars_new = drop_duplicate_stars(stars=stars_new)
        logger.debug("Dropped duplicate stars. Result:\n{stars}".format(stars=stars_new))
        stars_new.index.names = ['star_index']
        stars_new.columns.names = ['quantity_unit']
        if key == sorted_image_keys[0]:
            if len(stars_new) < 1:
                raise IOError(("The first frame must have stars:\n" +
                               "frame_tracking_number = {ftnum_new}\n" +
                               "stars =\n{stars}").format(stars=stars_new))
            timeseries_dict[ftnum_new] = stars_new
            timeseries_dict[ftnum_new]['matchedprev_bool'] = np.NaN
            logger.info(("Initial stars found.\n" +
                         "Frame tracking number: {ftnum}\n" +
                         "All current stars:\n" +
                         "{stars}").format(ftnum=ftnum_new, stars=timeseries_dict[ftnum_new]))
        else:
            # noinspection PyUnboundLocalVariable
            matched_stars = match_stars(image1=last_image_with_stars, image2=image_new, #@UndefinedVariable
                                        stars1=last_stars, stars2=stars_new) #@UndefinedVariable
            timeseries_dict[ftnum_new] = matched_stars['stars2']
            timeseries_dict[ftnum_new].rename(columns={'verif2to1': 'matchedprev_bool'}, inplace=True)
            logger.debug("Matched stars:\n{stars}".format(stars=timeseries_dict[ftnum_new]))
            # Report if new stars were found.
            if len(timeseries_dict[ftnum_new]) != len(timeseries_dict[last_ftnum_with_stars]): #@UndefinedVariable
                logger.info(("New stars found.\n" +
                             "Frame tracking number: {ftnum}\n" +
                             "All current stars:\n" +
                             "{stars}").format(ftnum=ftnum_new, stars=timeseries_dict[ftnum_new]))
        # Reset variables for next iteration, checking for clouds.
        if (key == sorted_image_keys[0]) or (len(stars_new) > 0):
            last_image_with_stars = image_new
            last_ftnum_with_stars = ftnum_new
            last_stars = timeseries_dict[ftnum_new][['x_pix', 'y_pix', 'sigma_pix']]
        # Do aperture photometry.
        # TODO: do median subtraction above and remove unnecessary subframing
        image_new -= np.nanmedian(image_new)
        positions = timeseries_dict[ftnum_new][['x_pix', 'y_pix']].values
        for radius in radii:
            apertures = photutils.CircularAperture(positions=positions, r=radius)
            # noinspection PyArgumentList
            phot_table = photutils.aperture_photometry(data=image_new, apertures=apertures)
            timeseries_dict[ftnum_new][('flux_ADU', radius)] = phot_table['aperture_sum']
        logger.debug("Calculated aperture photometry:\n{stars}".format(stars=timeseries_dict[ftnum_new]))
        # Record timestamp
        timestamps_dict[ftnum_new] = {'exposure_start_timestamp_UTC': dobj[key].meta['time_stamp_exposure_started'],
                                      'exposure_end_timestamp_UTC': dobj[key].meta['time_stamp_exposure_ended']}
        print_progress(key=key)
    # Format timeseries.
    timeseries = pd.concat(timeseries_dict, axis=0).stack()
    timeseries.index.names = ['frame_tracking_number', 'star_index', 'quantity_unit']
    timeseries = timeseries.unstack(['star_index', 'quantity_unit'])
    timeseries.sort_index(axis=1, inplace=True)
    # Format timestamps.
    footer_metadata = BeautifulSoup(dobj['footer_xml'], "xml")
    ts_begin = footer_metadata.find(name='TimeStamp', event='ExposureStarted').attrs['absoluteTime']
    dt_begin = dateutil.parser.parse(ts_begin)
    ticks_per_second = int(footer_metadata.find(name='TimeStamp', event='ExposureStarted').attrs['resolution'])
    timestamps = pd.DataFrame.from_dict(timestamps_dict, orient='index')
    timestamps.index.names = ['frame_tracking_number']
    timestamps.columns.names = ['quantity']
    timestamps['exposure_mid_timestamp_UTC'] = timestamps.mean(axis=1)
    timestamps = timestamps.applymap(lambda x: x / ticks_per_second)
    timestamps = timestamps.applymap(lambda x: dt_begin + dt.timedelta(seconds=x))
    return timestamps, timeseries

def plot_positions(timeseries, zoom=None, show_line_plots=True):
    """Make plots of star positions for all star indices.

    Parameters
    ----------
    timeseries : pandas.DataFrame
    zoom : {None}, optional, tuple
        x-y ranges of image for zoom. Must have format ((xmin, xmax), (ymin, ymax)).
    show_line_plots : {True}, optional, bool
        Show plots of (x_pix, y_pix, sigma_pix) vs frame_tracking_number for every star.

    Returns
    -------
    None

    See Also
    --------
    logger, make_timestamps_timeseries, plot_matches, plot_stars

    Notes
    -----
    SEQUENCE_NUMBER : 7.1

    """
    # If using zoom, check input.
    if zoom is not None:
        if not (len(zoom) == 2 and len(zoom[0]) == 2 and len(zoom[1]) == 2):
            raise IOError(("Required: Format of `zoom` must be ((xmin, xmax), (ymin, ymax)).\n" +
                           "zoom = {zoom}").format(zoom=zoom))
        ((xmin, xmax), (ymin, ymax)) = zoom
        if not (xmin >= 0 and ymin >= 0):
            raise IOError(("`zoom` = ((xmin, xmax), (ymin, ymax)). Required: xmin and ymin >= 0.\n" +
                           "zoom = {zoom}").format(zoom=zoom))
        if not ((xmax - xmin) >= 1 and (ymax - ymin) >= 1):
            raise IOError(("`zoom` = ((xmin, xmax), (ymin, ymax)). Required: (xmax - xmin) and (ymax - ymin) >= 1.\n" +
                           "zoom = {zoom}").format(zoom=zoom))
    sorted_star_idxs = sorted(timeseries.columns.levels[0].values)
    for star_idx in sorted_star_idxs:
        plt.scatter(x=timeseries[(star_idx, 'x_pix')], y=timeseries[(star_idx, 'y_pix')],
                    c=timeseries.index.values, s=50.0, cmap=plt.cm.jet, linewidths=0) #@UndefinedVariable
    plt.colorbar(ticks=np.linspace(timeseries.index.min(), timeseries.index.max(), 5, dtype=int))
    if zoom is not None:
        # noinspection PyUnboundLocalVariable
        plt.xlim(xmin=xmin, xmax=xmax)
        plt.xlabel('x_pix')
        # noinspection PyUnboundLocalVariable
        plt.ylim(ymin=ymin, ymax=ymax)
        plt.ylabel('y_pix')
    plt.gca().invert_yaxis()
    last_image_idx = timeseries.index.max()
    for star_idx in sorted_star_idxs:
        (x_pix, y_pix) = timeseries.loc[last_image_idx, (star_idx, ['x_pix', 'y_pix'])].values
        plt.annotate(str(star_idx), xy=(x_pix, y_pix), xycoords='data', xytext=(0, 0),
                     textcoords='offset points', color='black', fontsize=18, rotation=0)
    plt.title("Star positions by frame tracking number")
    plt.show()
    if show_line_plots:
        for star_idx in sorted_star_idxs:
            pd.DataFrame.plot(timeseries[star_idx][['x_pix', 'y_pix', 'sigma_pix']], kind='line',
                              secondary_y='sigma_pix', title="Star index: {idx}".format(idx=star_idx))
            plt.show()
    return None


def make_lightcurves(timestamps, timeseries, target_idx, comparison_idxs='all', ftnums_drop=None,
                    ftnums_norm=None, ftnums_calc_detrend=None, ftnums_apply_detrend=None, fixed_degree=None,
                    show_plots=False):
    """Make lightcurves from timestamps and timeseries.
    
    Optimal aperture radius is taken as ~1*FHWM, sec 5.4, [1]_. Median of each returned column with flux is 1.0.
    Median is taken after `ftnum_drop` and over `ftnum_med`.

    Parameters
    ----------
    timestamps : pandas.DataFrame
        Ouptut `timestamps` ``pandas.DataFrame`` from `make_timestamps_timeseries`.
    timeseries : pandas.DataFrame
        Ouptut `timeseries` ``pandas.DataFrame`` from `make_timestamps_timeseries`.
    target_idx : int
        ``int`` with star index to use as target star.
        Example: 0
    comparison_idxs : {'all'}, list, optional.
        ``list`` of ``int`` with star indices to use as comparison stars.
        If default 'all', uses all stars except for target as comparison stars. 
        Example: [2, 3, 4]
    ftnums_drop : {None}, list, optional
        ``list`` of ``int`` with frame tracking numbers of images to drop. Use for dropping frames due to clouds.
        If default ``None``, no frames are dropped.
        Example: ftnum_drop = range(10, 14) + range(20, 21) = [10, 11, 12, 13, 20]
    ftnums_norm : {None, 'all'}, list, optional
        ``list`` of ``int`` with frame tracking numbers of images to use for normalization.
        The entire lightcurve will be scaled so that median flux of `ftnums_norm` is 1.0.
        If default ``None``, lightcurve is not scaled.
        If 'all', entire lightcurve is scaled so that median flux of the entire lightcurve is 1.0.  
        Example: See example for `ftnum_drop`.
    ftnums_calc_detrend : {None, 'all'}, list, optional
        ``list`` of ``int`` with frame tracking numbers of images from which to calculate the polynomial for detrending.
        Requries that `ftnums_norm` is not ``None``. A warning is raised if `ftnums_norm` does not overlap with
        `ftnums_calc_detrend` since fluxes will be detrended to level of normalized flux. 
        See `ftnums_apply_detrend` to select the data to be detrended.
        If default ``None``, no curve is fit for detrending.
        If 'all', a curve is fit to entire lightcurve for detrending.
        Example: See example for `ftnum_drop`.
    ftnums_apply_detrend : {None, 'all'}, list, optional
        ``list`` of ``int`` with frame tracking numbers of images to which to apply the polynomial for detrending.
        Requries that `ftnums_calc_detrend` is not ``None``. A warning is raised if `ftnums_apply_detrend` is not a
        subset of `ftnums_calc_detrend` since the fit may not be relevant to data outside of the fit's domain .
        If default ``None``, no detrending is applied.
        if 'all', detrending is applied to entire lightcurve.
        Example: See example for `ftnum_drop`.
    fixed_degree : {None, 0, 1, 2, 3, 4}, int, optional
        Override degree of polynomial to fit for detrending.
        If default ``None``, polynomial degree is chosen by cross-validation and Bayesian Info. Crit. (see Notes below).
        Example: fixed_degree=2
    show_plots : {False}, bool, optional
        ``True``/``False`` flag to show diagnostic plots.

    Returns
    -------
    lightcurves : pandas.DataFrame
        Median of each returned column with flux is 1.0. Median is taken after `ftnum_drop` and over `ftnum_med`.
        Median calculation omits ``numpy.nan``.
        row names:
            `frame_tracking_number`: Frame tracking number from SPE metadata, >= 1. Example: 1
        column names:
            `quantity` : lightcurve quantity indexed by `frame_tracking_number`
        columns: 
            columns from `timestamps` ``pandas.DataFrame``
            `target_flux` : flux from target star
            `comparisons_sum_flux` : sum of fluxes from comparison stars
            `target_relative_flux` : `target_flux` / `comparisons_sum_flux`  
            if `ftnums_norm` is not None:
            `target_normalized_flux` : `target_flux` / median(`target_flux`)
            `comparisons_sum_normalized_flux` :  `comparisons_sum_flux` / median(`comparisons_sum_flux`)
            `target_relative_normalized_flux` : `target_relative_flux` / median(`target_relative_flux`)
            if `ftnums_apply_detrend` is not None:
            `target_relative_normalized_detrended_flux` : `target_relative_normalized_flux` with detrending applied

    See Also
    --------
    logger, make_timeseries_timestamps, plot_lightcurve

    Notes
    -----
    SEQUENCE_NUMBER : 8.0
    Data is undersampled if FHWM < 1.5 pix. sec 5.4 [1]_.
    Use `ftnums_calc_detrend` and `ftnums_apply_detrend` for correcting differential color extinction as a function
        of airmass. A polynomial is fit to `target_relative_normalized_flux`. If `fixed_degree` is default ``None``,
        the polynomial degree from 0 to 4 is selected using cross-validation and Bayesian Information Criterion
        (adapted from sec 8.11 of [2]_). Use `fixed_degree` when the statistical model of the polynomial fit to data
        should be informed by prior knowledge of a physical model. Differential photometric extinction is usually
        well-fit by a polynomial of degree 2. The polynomial coefficients are averaged over multiple fits. The data are
        detrended after being normalized by adding 1.0 to the residuals of the fit. 
    
    References
    ----------
    .. [1] Howell, 2009, "Handbook of CCD Astronomy"
    .. [2] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy"

    """
    # TODO: separate detrending function so that can apply multiple times.
    # TODO: add point rejection after making target_relative_flux (use method from astroml book, not sigma clipping)
    # Drop frames to be excluded (e.g. due to clouds).
    if ftnums_drop is None:
        logger.info("Not dropping any images.")
    else:
        logger.info("Dropping images with frame tracking numbers:\n{nums}".format(nums=ftnums_drop))
        timestamps = timestamps.drop(ftnums_drop, axis=0)
        timeseries = timeseries.drop(ftnums_drop, axis=0)
    # Select optimal aperture radius.
    # From Howell, 2009, sec 5.4, optimal aperture radius is ~1*FHWM. Data is undersampled if FHWM < 1.5 pix.
    # TODO: verify best aperture with scatter measure. Use SNR from photutils instead?
    # Example of decreasing scatter with radius:
    # comp_norm.apply(astroML.stats.sigmaG, axis=0).plot(marker='o', markersize=4)
    fwhm_med = \
        sigma_to_fwhm(
            np.nanmedian(
                [sigma for sigma in
                 timeseries.swaplevel('quantity_unit', 'star_index', axis=1)['sigma_pix'].values.flatten()
                 if sigma is not np.NaN]))
    radii = [label[1] for label in timeseries.columns.levels[1].values if len(label) == 2]
    best_radius = radii[np.nanargmin(np.abs(radii - fwhm_med))]
    logger.info("Optimal photometry aperture radius (pixels):\nbest_radius = {rad}".format(rad=best_radius))
    # Initialize lightcurves dataframe from timestamps.
    # Select target and comparison stars. Sum comparison star fluxes.
    # Define `exposure_mid_timestamp_UTC`, `target_flux`, `comparisons_sum_flux`, `target_relative_flux`
    lightcurves = timestamps.copy()
    lightcurves.index.names = ['frame_tracking_number']
    lightcurves.columns.names = ['quantity'] 
    logger.info("Target star index:\ntarget_idx = {idx}".format(idx=target_idx))
    lightcurves['target_flux'] = timeseries[(target_idx, ('flux_ADU', best_radius))]
    if comparison_idxs == 'all':
        comps = timeseries.drop(target_idx, level='star_index', axis=1)
        comp_idxs = np.delete(timeseries.columns.levels[0].values, target_idx)
    else:
        comps = timeseries.loc[:, comparison_idxs]
        comp_idxs = comparison_idxs
    logger.info("Comparison star indices:\ncomp_idxs = {idxs}".format(idxs=comp_idxs))
    if target_idx in comp_idxs:
        logger.warning("Target star is being used as a comparison star.")
    for comp_idx in comp_idxs:
        if comp_idx == comp_idxs[0]:
            lightcurves['comparisons_sum_flux'] = \
                comps[(comp_idx, ('flux_ADU', best_radius))]
        else:
            lightcurves['comparisons_sum_flux'] = \
                lightcurves['comparisons_sum_flux'].add(comps[(comp_idx, ('flux_ADU', best_radius))], fill_value=0.0)
    lightcurves['target_relative_flux'] = lightcurves['target_flux'].divide(lightcurves['comparisons_sum_flux'])
    # Normalize timeseries to get relative changes in brightness.
    # Ensure that all timeseries have median value of 1.0 (over chosen frames).
    if ftnums_norm is None:
        logger.info("Not normalizing lightcurves to median.")
    else:
        if ftnums_norm == 'all':
            logger.info("Normalizing lightcurves to median from all images.")
            ftnums_norm = lightcurves.index.values
        else:
            logger.info(("Normalizing lightcurves to median from images with frame tracking numbers:\n" +
                         "{nums}").format(nums=ftnums_norm))
        targ_median = lightcurves.loc[ftnums_norm, 'target_flux'].median(axis=0, skipna=True)
        lightcurves['target_normalized_flux'] = lightcurves['target_flux'].divide(targ_median)
        compsum_median = lightcurves.loc[ftnums_norm, 'comparisons_sum_flux'].median(axis=0, skipna=True)
        lightcurves['comparisons_sum_normalized_flux'] = lightcurves['comparisons_sum_flux'].divide(compsum_median)
        targrel_median = lightcurves.loc[ftnums_norm, 'target_relative_flux'].median(axis=0, skipna=True)
        lightcurves['target_relative_normalized_flux'] = lightcurves['target_relative_flux'].divide(targrel_median)
        # Detrend lightcurve to remove differential color extinction.
        # Calculate polynomial models for detrending using cross-validation and Bayesian Information Criterion.
        # Shuffle and split into training set (80%) and cross-validation set (20%).
        # Do once for each degree to determine best-fit number of degrees.
        if ftnums_calc_detrend is None:
            logger.info("Not calculating polynomial to detrend lightcurve.")
        else:
            if ftnums_calc_detrend == 'all':
                logger.info("Calculating polynomial to detrend lightcurve from all images.")
                ftnums_calc_detrend = lightcurves.index.values
            else:
                logger.info(("Calculating polynomial to detrend lightcurve from images with frame tracking numbers:\n" + 
                             "{nums}").format(nums=ftnums_calc_detrend))
            if not set(ftnums_norm) & set(ftnums_calc_detrend):
                logger.warning("`ftnums_norm` and `ftnums_calc_detrend` do not overlap.")
            num_calc_detrend = len(ftnums_calc_detrend)
            if num_calc_detrend < 5:
                raise IOError(("`ftnums_calc_detrend` must have at least 5 images:\n" +
                               "len(ftnums_calc_detrend) = {num}").format(num=num_calc_detrend))
            # Extract index and fluxes as numpy.ndarray to use "fancy indexing".
            ftnums_calc_detrend = lightcurves.loc[ftnums_calc_detrend, 'target_relative_normalized_flux'].index.values
            fluxes_calc_detrend = lightcurves.loc[ftnums_calc_detrend, 'target_relative_normalized_flux'].values
            degrees = range(5)
            models = pd.DataFrame(index=degrees, columns=['model', 'train_err', 'cval_err'])
            models.index.names = ['degree']
            models.columns.names = ['quantity']
            for deg in degrees:
                logger.debug("deg = {deg}".format(deg=deg))
                # Split data into training set and cross-validation set.
                for (idxs_train, idxs_cval) in sklearn_cval.ShuffleSplit(num_calc_detrend, n_iter=1, test_size=0.2, random_state=0):
                    (ftnums_train, fluxes_train) = (ftnums_calc_detrend[idxs_train], fluxes_calc_detrend[idxs_train])
                    (ftnums_cval, fluxes_cval) = (ftnums_calc_detrend[idxs_cval], fluxes_calc_detrend[idxs_cval])
                    model = np.polyfit(ftnums_train, fluxes_train, deg)
                    models.loc[deg, 'model'] = model
                    logger.debug("model = {mod}".format(mod=model))
                    train_err = np.sqrt(np.sum((np.polyval(model, ftnums_train) - fluxes_train)**2.0) / len(fluxes_train))
                    models.loc[deg, 'train_err'] = train_err
                    logger.debug("train_err = {terr}".format(terr=train_err))
                    cval_err = np.sqrt(np.sum((np.polyval(model, ftnums_cval) - fluxes_cval)**2.0) / len(fluxes_cval))
                    models.loc[deg, 'cval_err'] = cval_err
                    logger.debug("cval_err = {cerr}".format(cerr=cval_err))
            logger.debug("models =\n{mods}".format(mods=models))
            # Estimate intrinsic scatter for weighting BIC.
            scatter_est = models['cval_err'].min()
            logger.info(("Estimate for intrinsic scatter in lightcurve:\n" +
                         "scatter_est = {est}").format(est=scatter_est))
            models.loc[:, 'train_BIC'] = (np.sqrt(num_calc_detrend) * np.divide(models['train_err'], scatter_est)) + \
                np.multiply(degrees, np.log(num_calc_detrend))
            models.loc[:, 'cval_BIC'] = (np.sqrt(num_calc_detrend) * np.divide(models['cval_err'], scatter_est)) + \
                np.multiply(degrees, np.log(num_calc_detrend))
            # Calcualte best model with optimized number of degrees.
            # Do 5 times to mitigate influence of outliers then average.
            if fixed_degree is None:
                best_degree = models['cval_BIC'].idxmin()
                logger.info(("Optimal degree of polynomial model for detrending:\n" +
                             "best_degree = {deg}").format(deg=best_degree))
            else:
                best_degree = fixed_degree
                logger.info(("Fixed degree of polynomial model for detrending:\n" +
                             "best_degree = `fixed_degree` = {deg}").format(deg=best_degree))
            best_models = []
            for (idxs_train, idxs_cval) in sklearn_cval.ShuffleSplit(len(ftnums_calc_detrend), n_iter=5, test_size=0.2, random_state=0):
                (ftnums_train, fluxes_train) = (ftnums_calc_detrend[idxs_train], fluxes_calc_detrend[idxs_train])
                (ftnums_cval, fluxes_cval) = (ftnums_calc_detrend[idxs_cval], fluxes_calc_detrend[idxs_cval])
                model = np.polyfit(ftnums_train, fluxes_train, best_degree)
                best_models.append(model)
            best_model = np.mean(best_models, axis=0)
            logger.info(("Coefficients of optimal polynomial model for detrending:" +
                         "\nbest_model = {mod}").format(mod=best_model))
            # Apply best model to detrend lightcurve.
            if ftnums_apply_detrend is None:
                logger.info("Not applying polynomial to detrend lightcurve.")
            else:
                if ftnums_apply_detrend == 'all':
                    logger.info("Applying polynomial to detrend entire lightcurve.")
                    ftnums_apply_detrend = lightcurves.index.values
                else:
                    logger.info(("Applying polynomial to detrend lightcurve images with frame tracking numbers:\n" + 
                                 "{nums}").format(nums=ftnums_apply_detrend))
                if not set(ftnums_apply_detrend) <= set(ftnums_calc_detrend):
                    logger.warning("`ftnums_apply_detrend` is not a subset of `ftnums_calc_detrend`.")
                # Extract index and fluxes as numpy.ndarray to use "fancy indexing".
                ftnums_apply_detrend = lightcurves.loc[ftnums_apply_detrend, 'target_relative_normalized_flux'].index.values
                fluxes_apply_detrend = lightcurves.loc[ftnums_apply_detrend, 'target_relative_normalized_flux'].values
                fluxes_fit = np.polyval(best_model, ftnums_apply_detrend)
                fluxes_residual = fluxes_apply_detrend - fluxes_fit
                fluxes_detrended = fluxes_residual + 1.0
                lightcurves['target_relative_normalized_detrended_flux'] = lightcurves['target_relative_normalized_flux']
                lightcurves.loc[ftnums_apply_detrend, 'target_relative_normalized_detrended_flux'] = fluxes_detrended
    # Show diagnostic plots.
    if show_plots:
        df_plot = lightcurves[['target_flux',
                               'comparisons_sum_flux',
                               'target_relative_flux']]
        pd.DataFrame.plot(df_plot,
                          title="Absolute and relative lightcurves",
                          secondary_y=['target_relative_flux'],
                          marker='o', markersize=4, linestyle='')
        plt.show()
        if ftnums_norm is not None:
            df_norm = lightcurves.loc[ftnums_norm, ['target_relative_normalized_flux']].copy()
            df_norm.rename(columns={'target_relative_normalized_flux': 'ftnums_norm'}, inplace=True)
            df_plot = \
                pd.concat([lightcurves[['target_normalized_flux',
                                        'comparisons_sum_normalized_flux',
                                        'target_relative_normalized_flux']],
                           df_norm],
                          axis=1)
            pd.DataFrame.plot(df_plot,
                              title="Absolute and relative normalized lightcurves",
                              marker='o', markersize=4, linestyle='')
            plt.legend(framealpha=0.5)
            plt.show()
            if ftnums_calc_detrend is not None:
                pd.DataFrame.plot(models,
                                  title=("Detrending polynomial model selection:\n" +
                                         "Training and cross-validation error and Bayesian Info. Crit."),
                                  secondary_y=['train_BIC', 'cval_BIC'],
                                  marker='o')      
                plt.show()
                df_calc_detrend = lightcurves.loc[ftnums_calc_detrend, ['target_relative_normalized_flux']].copy()
                df_calc_detrend.rename(columns={'target_relative_normalized_flux': 'ftnums_calc_detrend'}, inplace=True)
                df_plot = \
                    pd.concat([lightcurves[['target_relative_normalized_flux']],
                               df_norm,
                               df_calc_detrend],
                              axis=1)
                pd.DataFrame.plot(df_plot,
                                  title=("Relative normalized lightcurve\n" +
                                         "with detrending model"),
                                  marker='o', markersize=4, linestyle='')
                fluxes_calc_detrend = np.polyval(best_model, ftnums_calc_detrend)
                plt.plot(ftnums_calc_detrend, fluxes_calc_detrend, c='red', linewidth=3)
                plt.legend(framealpha=0.5)
                plt.show()
                if ftnums_apply_detrend is not None:
                    df_apply_detrend = lightcurves.loc[ftnums_apply_detrend, ['target_relative_normalized_flux']].copy()
                    df_apply_detrend.rename(columns={'target_relative_normalized_flux': 'ftnums_apply_detrend'}, inplace=True)
                    df_plot = \
                        pd.concat([lightcurves[['target_relative_normalized_detrended_flux']],
                                   df_norm,
                                   df_calc_detrend,
                                   df_apply_detrend],
                                  axis=1)
                    pd.DataFrame.plot(df_plot,
                                      title=("Relative normalized detrended lightcurve\n" +
                                             "with detrending model"),
                                      marker='o', markersize=4, linestyle='')
                    plt.plot(ftnums_calc_detrend, fluxes_calc_detrend, c='red', linewidth=3)
                    plt.legend(framealpha=0.5)
                    plt.show()
    return lightcurves


def plot_lightcurve(lightcurves, col_timestamps='exposure_mid_timestamp_UTC', col_fluxes='target_relative_flux',
                    fpath=None, **kwargs):
    """Plot lightcurve.

    Parameters
    ----------
    lightcurves : pandas.DataFrame
        ``pandas.DataFrame`` output from `make_lightcurves`.
    col_timestamps : {'exposure_mid_timestamp_UTC'}, string, optional
        Column of `lightcurves` that is timestamps to be plotted on x-axis.
    col_fluxes : {'target_relative_flux'}, string, optional
        Column of `lightcurves` that is fluxes to be plotted on y-axis.
    fpath : {None}, string, optional
        Path to save file. Format can be '.pdf', '.eps', .jpg', '.png', etc. 
    kwargs : keyword arguments, optional
        keywords to pass to ``matplotlib.pyplot``
        Examples:
            ylim=[0.0, 1.0]
            {'ylim': [0.0, 1.0]}

    Returns
    -------
    None

    See Also
    --------
    make_lightcurves, logger, plot_positions, plot_matches, plot_stars

    Notes
    -----
    SEQUENCE_NUMBER : 8.1

    """
    plt.figure()
    lightcurve = lightcurves[[col_timestamps, col_fluxes]].set_index(keys=[col_timestamps])
    pd.DataFrame.plot(lightcurve,
                      title="{fpath}\n{ts}".format(fpath=os.path.basename(fpath),
                                                 ts=lightcurve.index[0].isoformat()),
                      legend=False, marker='o', markersize=2, linestyle='', **kwargs)
    plt.xlabel(col_timestamps)
    plt.ylabel(col_fluxes)
    if fpath is not None:
        logger.info("Writing plot to: {fpath}".format(fpath=fpath))
        plt.savefig(fpath, bbox_inches='tight')
    plt.show()
    return None
